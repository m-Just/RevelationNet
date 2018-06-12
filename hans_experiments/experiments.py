from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import numpy as np
import tensorflow as tf

from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import AccuracyReport, set_log_level

from cifar10_classifier import make_simple_cnn, make_resnet


def data_cifar10(train_start, train_end, test_start, test_end):

    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    import keras
    from keras.datasets import cifar10
    from keras.preprocessing.image import ImageDataGenerator

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

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    return datagen, (x_train, y_train), (x_test, y_test)

def train_classifier(model_name, nb_epochs):
    tf.set_random_seed(1822)
    report = AccuracyReport()
    set_log_level(logging.DEBUG)

    # Get CIFAR-10 data
    train_start = 0
    train_end = 50000
    test_start = 0
    test_end = 10000
    datagen, (x_train, y_train), (x_test, y_test) = \
        data_cifar10(train_start, train_end, test_start, test_end)

    #label_smooth = .1
    #y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Initialize model
    epoch_step = tf.Variable(0, trainable=False)
    if model_name == 'simple':
        model = make_simple_cnn()
        learning_rate = tf.constant(0.003)
    elif model_name == 'resnet':
        model = make_resnet(depth=32)
        learning_rate = tf.constant(0.001)
        '''
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step=epoch_step,
            decay_steps=nb_epochs, 
            decay_rate=0.9441,
            staircase=True)
        learning_rate = tf.case([
            (epoch_step > 180, lambda: 0.5e-6),
            (epoch_step > 160, lambda: 1e-6),
            (epoch_step > 120, lambda: 1e-5),
            (epoch_step > 80, lambda: 1e-4)],
            default=lambda: 1e-3)
        learning_rate = tf.case([
            (epoch_step > 70, lambda: 1e-5),
            (epoch_step > 50, lambda: 1e-4)],
            default=lambda: 1e-3)
        lr1 = tf.train.polynomial_decay(
            learning_rate,
            global_step=epoch_step,
            decay_steps=80,
            end_learning_rate=1e-4,
            power=0.5)
        lr1 = tf.train.exponential_decay(
            learning_rate,
            global_step=epoch_step,
            decay_steps=80,
            decay_rate=0.971628)
        lr2 = tf.train.exponential_decay(
            1e-4,
            global_step=epoch_step-80,
            decay_steps=120,
            decay_rate=0.944061)
        learning_rate = tf.cond(epoch_step < 80,
                                true_fn=lambda: lr1,
                                false_fn=lambda: lr2)
        '''
    else:
        raise ValueError()
    #for layer in model.layers:
    #    print(layer.name)
    preds = model.get_probs(x)

    batch_size = 32
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epoch_step': epoch_step # used for lr decay
    }
    rng = np.random.RandomState([2018, 6, 9])

    sess = tf.Session()

    def evaluate():
        eval_params = {'batch_size': 128}
        acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert x_test.shape[0] == test_end - test_start, x_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    dataflow = datagen.flow(x_train, y_train, batch_size=batch_size)
    model_train(sess, x, y, preds, x_train, y_train, dataflow=dataflow, 
                evaluate=evaluate, args=train_params, rng=rng,
                var_list=model.get_params())

    savedir = './tfmodels'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    model_savename = 'cifar10_%s_model_epoch%d' % (model_name, nb_epochs)
    saver.save(sess, os.path.join(savedir, model_savename))

    return report

def main(argv=None):
    train_classifier(model_name='resnet', nb_epochs=200)

if __name__ == '__main__':
    tf.app.run()
