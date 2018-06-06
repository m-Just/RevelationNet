import os
import random

import tensorflow as tf
import numpy as np

import data_loader

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
MODEL_NAME = 'keras_cifar10_trained_model'
CLASSIFIER_PATH = os.path.join(SAVE_DIR, MODEL_NAME)

def accuracy(pred, labels):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

def evaluate(sess, eval_acc, x, y_, adv_imgs, labels, target):
    print('Start evaluation...')
    n = len(adv_imgs)

    fidelity = 0.
    deceived = 0.
    for img, y in zip(adv_imgs, labels):
        fidelity += sess.run(eval_acc, feed_dict={x: img, y_: [y]})
        deceived += sess.run(eval_acc, feed_dict={x: img, y_: [target]})

    print('Fidelity rate on test set: %f' % (fidelity / n))
    print('Deceived rate on test set: %f' % (deceived / n))

def train_classifier():
    from classifiers import Classifier
    
    max_epoch = 50
    batch_size = 128
    imgsize = 32
    num_classes = 10
    data_augmentation = False

    if data_augmentation:
        pass
    else:
        (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).repeat().batch(128)
    iter_train = data_train.make_initializable_iterator()

    x = tf.placeholder(tf.float32, [batch_size, imgsize, imgsize, 3])
    y_= tf.placeholder(tf.float32, [batch_size, num_classes])

    with tf.variable_scope('conv') as scope:
        model = Classifier(x, expand_dim=False)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=y_))
    eval_acc = accuracy(model.logits, y_)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.01,
        momentum=0.9,
        use_nesterov=True)
    optim_step = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(iter_train.initializer)

    for n_epoch in range(max_epoch):
        for i in range(50000 / batch_size):
            batch = sess.run(iter_train.get_next())
            _, acc_val, loss_val = sess.run([optim_step, eval_acc, loss],
                feed_dict={x: batch[0], y_: batch[1]})
            if i % 10 == 0:
                print("Epoch: %d, Step: %d, Acc: %f, Loss: %f" % (n_epoch, i, acc_val, loss_val))

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

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    #model.save(CLASSIFIER_PATH)
    from keras.backend import get_session
    sess = get_session()
    saver = tf.train.Saver()
    saver.save(sess, CLASSIFIER_PATH)
    print('Saved trained model at %s ' % CLASSIFIER_PATH)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])

def attack():
    from FGSM import Generator

def noise_defense():
    from FGSM import Generator
    from classifiers import NoisyClassifier

    imgsize = 32
    num_classes = 10
    target_label = 0
    eps_val = 0.3
    num_samples = 200

    x = tf.Variable(tf.zeros([imgsize, imgsize, 3]))
    y_= tf.placeholder(tf.float32, [1, num_classes])

    with tf.variable_scope('conv') as scope:
        model = NoisyClassifier(x)
    pred = tf.nn.softmax(model.logits)
    eval_acc = accuracy(pred, y_)

    fgsm_agent = Generator(imgsize, x, model.logits)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.trainable_variables('conv'))
    saver.restore(sess, CLASSIFIER_PATH)

    adv_imgs = []
    y_real = []
    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
    
    indices = range(len(y_test))
    random.shuffle(indices)
    for i in range(num_samples):
        sample_x = x_test[indices[i]]
        sample_y = y_test[indices[i]]
        
        acc_val = sess.run(eval_acc, feed_dict={x: sample_x, y_: [sample_y]})
        if acc_val == 1 and np.argmax(sample_y) != target_label:
            adv_img = fgsm_agent.generate(sess, sample_x, target_label, eps_val=eps_val)
            adv_imgs.append(adv_img)
            y_real.append(sample_y)

    print('Generated adversarial images %d/%d' % (len(adv_imgs), num_samples))
    y_target = np.eye(num_classes)[target_label]
    evaluate(sess, eval_acc, x, y_, adv_imgs, y_real, y_target)

    print(len(adv_imgs))

if __name__ == '__main__':
    train_classifier()
    #noise_defense()
