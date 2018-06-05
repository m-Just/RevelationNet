import os    

import tensorflow as tf
import numpy as np

import data_loader

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
MODEL_NAME = 'keras_cifar10_trained_model'
CLASSIFIER_PATH = os.path.join(SAVE_DIR, MODEL_NAME)

def accuracy(pred, labels):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

def train_classifier():
    from cifar10_classifier import Classifier
    from keras.optimizers import SGD

    batch_size = 128
    num_classes = 10
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

def noise_defense():
    from FGSM import Generator
    from defense import NoisyClassifier

    imgsize = 32
    cls_num = 10
    target_label = 0
    eps_val = 0.3
    test_sample = 100

    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()

    x = tf.Variable(tf.zeros([imgsize, imgsize, 3]))
    y_= tf.placeholder(tf.float32, [1, cls_num])

    with tf.variable_scope('conv') as scope:
        model = NoisyClassifier(x)
    pred = tf.nn.softmax(model.logits)
    fgsm_agent = Generator(imgsize, x, model.logits)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.trainable_variables('conv'))
    saver.restore(sess, CLASSIFIER_PATH)

    adv_imgs = []
    y_real = []
    for i, sample in enumerate(zip(x_test, y_test)):
        sample_x, sample_y = sample
        acc = sess.run(accuracy(pred, y_), feed_dict={x: sample_x, y_: [sample_y]})
        if acc == 1 and np.argmax(sample_y) != target_label:
            adv_img = fgsm_agent.generate(sess, sample_x, target_label, eps_val=eps_val)
            adv_imgs.append(adv_img)
            y_real.append(sample_y)
        #if i >= test_sample: break 

    adv_imgs = np.stack(adv_imgs, axis=0)
    print(adv_imgs.shape)

if __name__ == '__main__':
    #train_classifier()
    noise_defense()
