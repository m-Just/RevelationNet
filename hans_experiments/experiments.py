import logging

import numpy as np
import tensorflow as tf

from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import AccuracyReport, set_log_level

from cifar10_classifier import make_classifier

tf.set_random_seed(1822)
report = AccuracyReport()
set_log_level(logging.DEBUG)

def data_cifar10(train_start, train_end, test_start, test_end):

    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    import keras
    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[train_start:train_end]
    y_train = y_train[train_start:train_end]
    x_test = x_test[test_start:test_end]
    y_test = y_test[test_start:test_end]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

def train_classifier()
    # Get CIFAR-10 data
    train_start = 0
    train_end = 50000
    test_start = 0
    test_start = 10000
    (x_train, y_train), (x_test, y_test) = \
        data_cifar10(train_start, train_end, test_start, test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Initialize model
    model = make_classifier()
    preds = model.get_probs(x)

    train_params = {
        'nb_epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.01 
    }
    rng = np.random.RandomState([2018, 6, 9])

    sess = tf.Session()

    def evaluate():
        eval_params = {'batch_size': 128}
        acc = model_eval(sess, x, y, x_test, y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert x_test.shape[0] == test_end - test_start, x_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    model_train(sess, x, y, preds, x_train, y_train, evaluate=evaluate,
                args=train_params, rng=rng, var_list=model.get_params())
