import os
import numpy as np
from keras import Input, Model, optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, LSTM
import keras.backend as K

from attention_decoder import AttentionDecoder, loss_fn, accuracy_fn


def get_model(x_train: np.ndarray,
              y_train: np.ndarray,
              x_validate: np.ndarray = None,
              y_validate: np.ndarray = None,
              model_weights_path: str = 'data/Model/weights') -> Model:
    # create model
    num_encoder_tokens = x_train.shape[2]
    num_decoder_tokens = y_train.shape[2]

    # encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_outputs_1 = Bidirectional(LSTM(256, return_sequences=True))(encoder_inputs)
    encoder_outputs_2 = Bidirectional(LSTM(256, return_sequences=True))(encoder_outputs_1)
    encoder_outputs_3 = Bidirectional(LSTM(256, return_sequences=True))(encoder_outputs_2)
    encoder_outputs = encoder_outputs_3
    # decoder
    decoder_truths = Input(shape=(None,))
    outputs = AttentionDecoder(512, num_decoder_tokens, num_decoder_tokens)((encoder_outputs, decoder_truths))
    # compile
    model = Model([encoder_inputs, decoder_truths], outputs)
    adam = optimizers.adam_v2.Adam(learning_rate=3e-4)
    model.compile(optimizer=adam, loss=loss_fn, metrics=[accuracy_fn])
    print(model.summary())
    print("Input: {}".format(x_train.shape))
    print("Output: {}".format(y_train.shape))
    # callback
    checkpoint = ModelCheckpoint(model_weights_path, monitor='val_accuracy_fn', verbose=1,
                                 save_best_only=True, save_weights_only=True)

    if os.path.exists(model_weights_path + ".index"):
        model.load_weights(model_weights_path)
    else:
        # train
        try:
            x_train_truths = K.argmax(y_train)
            x_validate_truths = K.argmax(y_validate)
            history = model.fit([x_train, x_train_truths], y_train,
                                validation_data=([x_validate, x_validate_truths], y_validate),
                                batch_size=16,
                                epochs=32,
                                callbacks=[checkpoint])
        except KeyboardInterrupt as e:
            print('Model training stopped early.')
        # save
        model.save_weights("data/Model/weights_last")
        # model.save("data/Model/model")

    print("Model Done!")

    return model
