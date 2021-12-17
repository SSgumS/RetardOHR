import numpy as np
import random

from data_load import load_data
from model import get_model
from evaluate import evaluate

random.seed(0)

# load dataset
dataset, dataset_chars = load_data()
# train_size = tf.cast(len(X_train) / 10, dtype='int32')
# test_size = tf.cast(len(X_test) / 10, dtype='int32')
# dataset = pd.Series([X_train[:train_size], Y_train[:train_size], X_test[:test_size], Y_test[:test_size]])
# dataset.to_pickle('data/dataset.pkl.compressed_less', compression="gzip")
X, Y = dataset.values

size = X.shape[0]
X_train, X_validate, X_test = np.split(X, [int(.7 * size), int(.85 * size)])
Y_train, Y_validate, Y_test = np.split(Y, [int(.7 * size), int(.85 * size)])

# create model
model = get_model(X_train, Y_train, X_validate, Y_validate)


# evaluate model
chars_index = list(dataset_chars.index)
evaluate(X_test, Y_test, chars_index, model)
