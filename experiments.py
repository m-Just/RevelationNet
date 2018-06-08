import os
import random

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data_loader

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
CLASSIFIER_PATH = os.path.join(SAVE_DIR, 'new_model')
PRETRAINED_PATH = os.path.join(SAVE_DIR, 'pretrained_model')

def visualize(gridsize, imgs):
    fig = plt.figure(figsize=gridsize[::-1])
    gs = gridspec.GridSpec(*gridsize)
    gs.update(wspace=0.05, hspace=0.05)

    assert len(imgs) == gridsize[0] * gridsize[1]
    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)

    return fig

def accuracy(pred, labels):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

def evaluate(sess, assign_op, eval_acc, x, y_, adv_imgs, labels, target):

def train_classifier():
    from classifiers import Classifier
    
    max_epoch = 50
    batch_size = 128
    imgsize = 32
    weight_decay = 0 # disabled
    num_classes = 10
    data_augmentation = False

    print('WARNING: data augmentation not implemented. ' + \
        'For better model performance, please use train_keras_classifier instead')
    response = raw_input('Do you wish to continue? (y/N)')
    if response.lower() not in ['y', 'yes']: return

    if data_augmentation:
        pass
    else:
        (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_trian = data_train.shuffle(50000).repeat().batch(128)
    iter_train = data_train.make_initializable_iterator()

    x = tf.placeholder(tf.float32, [batch_size, imgsize, imgsize, 3])
    y_= tf.placeholder(tf.float32, [batch_size, num_classes])

    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    with tf.variable_scope('conv') as scope:
        model = Classifier(x, regularizer, expand_dim=False)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=y_))
    reg_loss = tf.losses.get_regularization_loss()
    loss += reg_loss

    eval_acc = accuracy(model.logits, y_)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.01,
        momentum=0.9,
        use_nesterov=True)
    optim_step = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(iter_train.initializer)

    next_batch = iter_train.get_next()
    for n_epoch in range(max_epoch):
        for i in range(50000 / batch_size):
            batch = sess.run(next_batch)
            _, acc_val, loss_val = sess.run([optim_step, eval_acc, loss],
                feed_dict={x: batch[0], y_: batch[1]})
            if i % 100 == 0:
                print("Epoch: %d, Step: %d, Acc: %f, Loss: %f" % (n_epoch, i, acc_val, loss_val))
        acc_avg = loss_avg = 0
        test_batch_num = len(y_test) / batch_size

        # validate on test set
        for i in range(test_batch_num):
            acc_val, loss_val = sess.run([eval_acc, loss],
                feed_dict={x: x_test[i*batch_size:(i+1)*batch_size],
                           y_: y_test[i*batch_size:(i+1)*batch_size]})
            acc_avg += acc_val
            loss_avg += loss_val
        print('Test accuracy: %f, loss: %f' % (acc_avg / test_batch_num, loss_avg / test_batch_num))

    saver = tf.train.Saver()
    saver.save(sess, CLASSIFIER_PATH)
    print('Saved trained model at %s ' % CLASSIFIER_PATH)

def train_keras_classifier():
    from cifar10_classifier import Classifier
    from keras.optimizers import SGD

    batch_size = 128
    epochs = 50
    data_augmentation = True

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    with tf.variable_scope('conv') as scope:
        model = Classifier().model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

    # load data and start training
    if data_augmentation:
        print('Using real-time data augmentation.')
        datagen, (x_train, y_train), (x_test, y_test) = data_loader.load_augmented_data()
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_test, y_test),
            workers=4)
    else:
        print('Not using data augmentation.')
        (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
        model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)

    # save as tensorflow model
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    model.save(CLASSIFIER_PATH)

    from keras.backend import get_session
    sess = get_session()
    saver = tf.train.Saver()
    saver.save(sess, PRETRAINED_PATH)
    print('Saved trained model at %s ' % PRETRAINED_PATH)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])

def attack_classifier(_Classifier, target_label,
    imgsize=32,
    num_classes=10,
    num_samples=100,
    eps_val=0.3,
    visualize_results=True):

    from FGSM import Generator

    x = tf.placeholder(tf.float32, [imgsize, imgsize, 3])
    x_adv = tf.Variable(tf.zeros([imgsize, imgsize, 3]))
    y_= tf.placeholder(tf.float32, [1, num_classes])

    assign_op = tf.assign(x_adv, x)
    with tf.variable_scope('conv') as scope:
        model = _Classifier(x_adv)
    eval_acc = accuracy(model.logits, y_)

    fgsm_agent = Generator(imgsize, x_adv, model.logits)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.trainable_variables('conv'))
    saver.restore(sess, PRETRAINED_PATH)

    adv_imgs = []
    y_real = []
    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()

    indices = range(len(y_test))
    random.shuffle(indices)

    for i in range(num_samples):
        sample_x = x_test[indices[i]]
        sample_y = y_test[indices[i]]
        
        sess.run(assign_op, feed_dict={x: sample_x})
        acc_val = sess.run(eval_acc, feed_dict={x: sample_x, y_: [sample_y]})
        if acc_val == 1 and np.argmax(sample_y) != target_label:
            adv_img = fgsm_agent.generate(sess, sample_x, target_label, eps_val=eps_val)
            adv_imgs.append(adv_img[-1])
            y_real.append(sample_y)

            if visualize_results:
                fig = visualize((10, 10), [sample_x] + adv_img[:99]) 
                plt.savefig('visualizations/%d.png' % i)
                plt.close(fig)

    print('Generated adversarial images %d/%d' % (len(adv_imgs), num_samples))
    y_target = np.eye(num_classes)[target_label]

    print('Start evaluation...')
    n = len(adv_imgs)

    fidelity = 0.
    deceived = 0.
    for img, y in zip(adv_imgs, labels):
        sess.run(assign_op, feed_dict={x: img})
        fidelity += sess.run(eval_acc, feed_dict={x: img, y_: [y]})
        deceived += sess.run(eval_acc, feed_dict={x: img, y_: [target]})

    print('Fidelity rate on test set: %f' % (fidelity / n))
    print('Deceived rate on test set: %f' % (deceived / n))

if __name__ == '__main__':
    #train_classifier()
    #train_keras_classifier()

    from classifiers import *
    attack_classifier(Classifier, target_label=0)
    #attack_classifier(NoisyClassifier, target_label=0)
