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
        learning_rate = tf.constant(0.001)
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
        '''
        lr1 = tf.train.polynomial_decay(
            learning_rate,
            global_step=epoch_step,
            decay_steps=80,
            end_learning_rate=1e-4,
            power=0.5)
        lr2 = tf.train.exponential_decay(
            1e-4,
            global_step=epoch_step-80,
            decay_steps=1,
            decay_rate=0.944061)
        learning_rate = tf.cond(epoch_step < 80,
                                true_fn=lambda: lr1,
                                false_fn=lambda: lr2)
        '''
        '''
    else:
        raise ValueError()
    assert len(model.get_params()) == len(tf.trainable_variables())
    #for layer in model.layers:
    #    print(layer.name)
    preds = model.get_probs(x)

    batch_size = 32
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epoch_step': epoch_step, # used for lr decay
        'weight_decay': 1e-4
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
    a = sess.run(tf.trainable_variables()[-5])
    print(a)

    savedir = './tfmodels'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    model_savename = 'cifar10_%s_model_epoch%d' % (model_name, nb_epochs)
    saver.save(sess, os.path.join(savedir, model_savename))

    return report

def attack_classifier(model_name, model_savepath, attack_method='fgsm', target=None, nb_samples=128):
    tf.set_random_seed(1822)
    report = AccuracyReport()
    set_log_level(logging.DEBUG)

    # Get CIFAR-10 data
    train_start = 0
    train_end = 50000
    test_start = 0
    test_end = 10000
    assert nb_samples <= test_end - test_start
    datagen, (x_train, y_train), (x_test, y_test) = \
        data_cifar10(train_start, train_end, test_start, test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Initialize model
    if model_name == 'simple':
        model = make_simple_cnn()
    elif model_name == 'simple_noisy':
        model = make_simple_cnn(noisy_linear=True)
    elif model_name == 'resnet':
        model = make_resnet(depth=32)
    else:
        raise ValueError()

    sess = tf.Session()
    saver = tf.train.Saver(var_list=model.get_params())
    saver.restore(sess, model_savepath)

    # Make sure the model load properly by running it against the test set
    preds = model.get_probs(x)
    eval_args = {'batch_size': 128}
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_args)
    print('Test accuracy on legitimate examples: %.4f' % acc)

    # Initiate attack
    if attack_method == 'fgsm':
        from cleverhans.attacks import FastGradientMethod
        method = FastGradientMethod(model, sess=sess)
        params = {'eps': 0.3,
                  'clip_min': 0.,
                  'clip_max': 1.}

    elif attack_method == 'basic_iterative':
        from cleverhans.attacks import BasicIterativeMethod
        method = BasicIterativeMethod(model, sess=sess)
        params = {'eps': 0.3,
                  'eps_iter': 0.02,
                  'nb_iter': 100,
                  'clip_min': 0.,
                  'clip_max': 1.}
        if target is not None:
            y_target = np.repeat(np.eye(10)[target:target+1], nb_samples, axis=0)
            params['y_target'] = tf.constant(y_target)

    adv_x = method.generate(x, **params)
    preds_adv = model.get_probs(adv_x)

    indices = range(nb_samples)
    rng = np.random.RandomState([2018, 6, 9])
    rng.shuffle(indices)
    x_sample = np.stack([x_test[indices[i]] for i in range(nb_samples)])
    y_sample = np.stack([y_test[indices[i]] for i in range(nb_samples)])

    eval_par = {'batch_size': min(nb_samples, 128)}
    acc = model_eval(sess, x, y, preds_adv, x_sample, y_sample, args=eval_par)
    print('Test accuracy on adversarial examples: %.4f' % acc)
    report.clean_train_adv_eval = acc
    if target is not None:
        acc = model_eval(sess, x, y, preds_adv, x_sample, y_target, args=eval_par)
        print('Success rate of targeted attacks on adversarial examples: %.4f' % acc)

    return report

def main(argv=None):
    #train_classifier(model_name='resnet', nb_epochs=200)
    #train_classifier(model_name='simple', nb_epochs=50)

    attack_classifier('simple_noisy', './tfmodels/cifar10_simple_model_epoch50',
                      attack_method='basic_iterative',
                      target=0)

if __name__ == '__main__':
    tf.app.run()
