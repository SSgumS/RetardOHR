import keras.losses
from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.engine.input_layer import Input
from keras.engine.input_spec import InputSpec
from keras.layers import AbstractRNNCell
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import RNN
from keras.models import Model
from keras.utils import tf_utils, losses_utils
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # maybe below is more clear implementation compared to older keras
    # at least it works the same for tensorflow, but not tested on other backends
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    return x


class AttentionDecoderCell(AbstractRNNCell):
    def __init__(self,
                 units,
                 output_size,
                 embedding_dim=32,
                 is_monotonic=False,
                 normalize_energy=False,
                 return_probabilities=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.start_token = output_size

        self.units = units
        self._output_size = output_size

        self.is_monotonic = is_monotonic
        self.normalize_energy = normalize_energy
        self.return_probabilities = return_probabilities

        self.embedding_dim = embedding_dim
        self.embedding_sublayer = Embedding(output_size + 1, embedding_dim)

        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoderCell, self).__init__(**kwargs)

        self.batch_size = None
        self.timesteps = None
        self.input_dim = None
        self._state_size = None
        self.x_seq = None
        self.y_true = None
        self.uses_learning_phase = True

    def add_scalar(self, initial_value=0, name=None, trainable=True):
        scalar = K.variable(initial_value, name=name)
        if trainable:
            self._trainable_weights.append(scalar)
        else:
            self._non_trainable_weights.append(scalar)
        return scalar

    @tf_utils.shape_type_conversion
    def build(self, input_shapes):
        """
            Embedding matrix for y (outputs)
        """
        self.embedding_sublayer.build(None)

        """
            Matrices for creating the context vector
        """
        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.embedding_dim, self.units),
                                   name='W_r',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_r = self.add_weight(shape=(self.units,),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.embedding_dim, self.units),
                                   name='W_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_z = self.add_weight(shape=(self.units,),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.embedding_dim, self.units),
                                   name='W_p',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_p = self.add_weight(shape=(self.units,),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_size),
                                   name='C_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_size),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.embedding_dim, self.output_size),
                                   name='W_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_o = self.add_weight(shape=(self.output_size,),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self._uxpb = None

        if self.is_monotonic:
            self.Energy_r = self.add_scalar(initial_value=-1,
                                            name='r')
        if self.normalize_energy:
            self.Energy_g = self.add_scalar(initial_value=1,
                                            name='g')

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if isinstance(inputs, list):
            assert len(inputs) == 2  # inputs == [encoder_outputs, y_true]
            encoder_outputs = inputs[0]
        else:
            encoder_outputs = inputs

        memory_shape = tf.shape(encoder_outputs)

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(encoder_outputs[:, 0], self.W_s))
        y0 = tf.fill((memory_shape[0],), np.int64(self.start_token))
        t0 = tf.fill((memory_shape[0],), np.int64(0))

        initial_states = [y0, s0, t0]
        if self.is_monotonic:
            # initial attention has form: [1, 0, 0, ..., 0] for each sample in batch
            alpha0 = K.ones((memory_shape[0], 1))
            alpha0 = K.switch(K.greater(memory_shape[1], 1),
                              lambda: K.concatenate([alpha0, K.zeros((memory_shape[0], memory_shape[1] - 1))], axis=-1),
                              alpha0)
            # like energy, attention is stored in shape (samples, time, 1)
            alpha0 = K.expand_dims(alpha0, -1)
            initial_states.append(alpha0)

        return initial_states

    def call(self, inputs, states, training=None, use_teacher_forcing=True):
        if self.is_monotonic:
            ytm, stm, timestep, previous_attention = states
        else:
            ytm, stm, timestep = states

        ytm = self.embedding_sublayer(ytm)

        if self.recurrent_dropout is not None and 0. < self.recurrent_dropout < 1.:
            stm = K.in_train_phase(K.dropout(stm, self.recurrent_dropout), stm)
            ytm = K.in_train_phase(K.dropout(ytm, self.recurrent_dropout), ytm)

        et = self._compute_energy(stm)

        if self.is_monotonic:
            at = self._compute_probabilities(et, previous_attention)
        else:
            at = self._compute_probabilities(et)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)

        # ~~~> calculate new hidden state

        # first calculate the "r" gate:
        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1 - zt) * stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(st, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if use_teacher_forcing:
            ys = K.in_train_phase(self.y_true[:, timestep[0]], K.argmax(yt, axis=-1))
            ys = K.flatten(ys)
        else:
            ys = K.flatten(K.argmax(yt, axis=-1))

        if self.return_probabilities:
            output = at
        else:
            output = yt

        next_states = [ys, st, timestep + 1]
        if self.is_monotonic:
            next_states.append(at)

        return output, next_states

    def _compute_energy(self, stm):
        # "concat" energy function
        # energy_i = g * V / |V| * tanh([stm, h_i] * W + b) + r
        _stm = K.dot(stm, self.W_a)

        V_a = self.V_a
        if self.normalize_energy:
            V_a = self.Energy_g * K.l2_normalize(self.V_a)

        et = K.dot(activations.tanh(K.expand_dims(_stm, axis=1) + self._uxpb),
                   K.expand_dims(V_a))

        if self.is_monotonic:
            et += self.Energy_r

        return et

    def _compute_probabilities(self, energy, previous_attention=None):
        if self.is_monotonic:
            # add presigmoid noise to encourage discreteness
            sigmoid_noise = K.in_train_phase(1., 0.)
            noise = K.random_normal(K.shape(energy), mean=0.0, stddev=sigmoid_noise)
            # encourage discreteness in train
            energy = K.in_train_phase(energy + noise, energy)

            p = K.in_train_phase(K.sigmoid(energy),
                                 K.cast(energy > 0, energy.dtype))
            p = K.squeeze(p, -1)
            p_prev = K.squeeze(previous_attention, -1)
            # monotonic attention function from tensorflow
            at = K.in_train_phase(
                tfa.seq2seq.monotonic_attention(p, p_prev, 'parallel'),
                tfa.seq2seq.monotonic_attention(p, p_prev, 'hard'))
            at = K.expand_dims(at, -1)
        else:
            # softmax
            at = activations.softmax(energy, axis=1)

        return at

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def get_config(self):
        config = {
            'units':
                self.units,
            'embedding_dim':
                self.embedding_dim,
            'return_probabilities':
                self.return_probabilities,
            'is_monotonic':
                self.is_monotonic,
            'normalize_energy':
                self.normalize_energy,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AttentionDecoderCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionDecoder(RNN):
    def __init__(self,
                 units,
                 output_size,
                 embedding_dim=32,
                 is_monotonic=False,
                 normalize_energy=False,
                 return_probabilities=False,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=True,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 time_major=False,
                 unroll=False,
                 **kwargs):
        cell_kwargs = {}
        cell = AttentionDecoderCell(
            units,
            output_size,
            embedding_dim=embedding_dim,
            is_monotonic=is_monotonic,
            normalize_energy=normalize_energy,
            return_probabilities=return_probabilities,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)

        super(AttentionDecoder, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)

    def build(self, input_shapes):
        if isinstance(input_shapes, (list, tuple)):
            input_shape = input_shapes[0]
        else:
            input_shape = input_shapes
        self.cell.batch_size, self.cell.timesteps, self.cell.input_dim = input_shape

        self.cell._state_size = [1, self.units, 1]
        if self.is_monotonic:
            self.cell._state_size.append((self.timesteps, 1))
        self.state_spec = [
            InputSpec(shape=[None] + tf.TensorShape(dim).as_list())
            for dim in self.state_size
        ]

        self.states = [None, None, None]  # y, s, t
        if self.is_monotonic:
            self.states.append(None)

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        super(AttentionDecoder, self).build(input_shapes)

        self.built = True

    def call(self, inputs, mask=None, training=None, initial_state=None, use_teacher_forcing=True):
        if isinstance(inputs, (list, tuple)):
            # teacher forcing for training
            self.cell.x_seq, self.cell.y_true = inputs
            self.cell.y_true = K.cast(self.y_true, dtype='int64')
            self.use_teacher_forcing = use_teacher_forcing
            inputs = K.expand_dims(self.y_true)
        else:
            # inference
            self.cell.x_seq = inputs
            self.use_teacher_forcing = False

        kwargs = {
            'training': training,
            'use_teacher_forcing': self.use_teacher_forcing}

        def step(inputs, states):
            return self.cell(inputs, states, **kwargs)

        # apply a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # therefore we can save computation time:
        self.cell._uxpb = _time_distributed_dense(self.x_seq, self.cell.U_a, b=self.cell.b_a,
                                                  dropout=self.cell.dropout,
                                                  input_dim=self.cell.input_dim,
                                                  timesteps=self.cell.timesteps,
                                                  output_dim=self.cell.units,
                                                  training=training)

        if initial_state is None or len(initial_state) != len(self.states):
            initial_state = self.cell.get_initial_state(self.x_seq)

        last_output, outputs, states = K.rnn(
            step,
            inputs=inputs,
            initial_states=initial_state
        )

        return outputs

    @property
    def units(self):
        return self.cell.units

    @property
    def is_monotonic(self):
        return self.cell.is_monotonic

    @property
    def x_seq(self):
        return self.cell.x_seq

    @property
    def timesteps(self):
        return self.cell.timesteps

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def y_true(self):
        return self.cell.y_true

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AttentionDecoder, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     return cls(**config)


def loss_fn(y_true, y_pred):
    crossentropy = keras.losses.CategoricalCrossentropy()

    # Mask padding values, they do not have to compute for loss
    mask = tf.math.logical_not(K.equal(K.argmax(y_true), 0))
    mask = tf.cast(mask, dtype=tf.int64)
    # Calculate the loss value
    loss = crossentropy(y_true, y_pred, sample_weight=mask)

    return loss


def accuracy_fn(y_true, y_pred):
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)
    correct = K.cast(K.equal(y_true, y_pred), dtype='float32')

    # 0 is padding, don't include those
    mask = K.cast(K.greater(y_true, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)

    return n_correct / n_total


if __name__ == '__main__':
    from keras.layers import Bidirectional, LSTM
    from data.reader import Data, Vocabulary
    from utils.examples import run_examples

    input_vocab = Vocabulary(
        'data/human_vocab.json', padding=32)
    output_vocab = Vocabulary(
        'data/machine_vocab.json', padding=32)

    training = Data('data/training.csv', input_vocab, output_vocab)
    validation = Data('data/validation.csv', input_vocab, output_vocab)
    training.load()
    validation.load()
    training.transform()
    validation.transform()
    print('Datasets Loaded.')

    n_input = input_vocab.size()
    n_output = output_vocab.size()

    inputs = Input(shape=(None,), dtype='int32')
    outputs_truth = Input(shape=(None,), dtype='int32')

    inputs_embed = Embedding(n_input, n_input, weights=[np.eye(n_input)], trainable=False)(inputs)

    rnn_encoded = Bidirectional(LSTM(128, return_sequences=True))(inputs_embed)

    attention_decoder = AttentionDecoder(128, n_output)
    outputs = attention_decoder([rnn_encoded, outputs_truth])

    model = Model(inputs=[inputs, outputs_truth], outputs=[outputs])
    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    try:
        model.fit_generator(generator=training.generator(64),
                            steps_per_epoch=100,
                            validation_data=validation.generator(64),
                            validation_steps=100,
                            epochs=64)
    except KeyboardInterrupt as e:
        print('Model training stopped early.')

    run_examples(model, input_vocab, output_vocab)
