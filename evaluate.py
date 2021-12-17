import numpy as np
from keras import Model
import keras.backend as K

eos_char = "\u0003"


def evaluate(x: np.ndarray,
             y: np.ndarray,
             chars_index: list,
             model: Model):
    max_chars = y.shape[1]

    def decode_seq(seq):
        decoded_sentence = ""
        for c_i in seq:
            c = chars_index[c_i]

            # Exit condition: either hit max length or find stop character.
            if c == eos_char or len(decoded_sentence) == max_chars:
                break

            decoded_sentence += c
        return decoded_sentence.strip()

    test_inputs = [x, np.zeros(K.argmax(y).shape)]
    scores = model.evaluate(test_inputs, y)
    predict = model.predict(test_inputs)

    total = len(x)
    correct = 0
    for i in range(total):
        truth = decode_seq(K.argmax(y[i]))
        pred = decode_seq(K.argmax(predict[i]))
        print("{0}, {1}".format(truth, pred))
        if truth == pred:
            correct += 1

    print("Evaluation Done: {0:.2f} (Word), {1:.2f} (Character)".format(100 * correct / total, 100 * scores[1]))
