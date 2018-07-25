import time

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from classifiers import Classifier
import data_loader
from gradient_attack import Generator

PRETRAINED_PATH = '../saved_models/pretrained_model'

def build_graph_init(x, sess, expand_dim=False, output_units=10, pretrained=None):

    with tf.variable_scope('conv'):
        model = Classifier(x, expand_dim=expand_dim, output_units=output_units)
    sess.run(tf.global_variables_initializer())

    if pretrained is not None:
        var_list = [var for var in tf.trainable_variables() if var.op.name.startswith('conv')]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, pretrained)

        print('Restored variables:')
        for var in var_list:
            print(var.op.name)
    
    return model

def load_adversarial_samples():

    x_adv = []
    for i in range(50000):
        img = plt.imread('./adv_imgs/%d.png' % i)[..., :3]
        assert np.min(img) >= 0. and np.max(img) <= 1.
        x_adv.append(img)
    y_adv = np.repeat([np.eye(11)[-1]], len(x_adv), axis=0)
    return x_adv, y_adv

def eval_acc(logits, labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

def validate(sess, accuracy, x, y_, x_test, y_test):
    acc_val = 0.
    num_imgs = len(x_test)
    num_batches = num_imgs / 100
    for n_batch in range(num_batches):
        x_batch = x_test[n_batch * 100 : (n_batch + 1) * 100]
        y_batch = y_test[n_batch * 100 : (n_batch + 1) * 100]
        batch_acc_val = sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch})
        acc_val += batch_acc_val
    acc_val /= num_batches
    print('Test accurcacy on legitimate images: %f' % acc_val)

def evaluate_perturbation(p):
    l_one = np.sum(np.abs(p))
    l_two = np.sqrt(np.sum(p ** 2))
    l_inf = np.max(np.abs(p))
    print('L-one: %.2f' % l_one)
    print('L-two: %.2f' % l_two)
    print('L-inf: %.2f' % l_inf)
    
def adv_train():

    # load training and testing data
    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
    y_train = np.concatenate((y_train, np.zeros([len(y_train), 1], dtype=y_train.dtype)), axis=1)
    y_test = np.concatenate((y_test, np.zeros([len(y_test), 1], dtype=y_test.dtype)), axis=1)
    x_adv, y_adv = load_adversarial_samples()

    for i in range(50000):
        linf = np.max(np.abs(x_adv[i] - x_train[i]))
        assert linf < 0.2 + 1e-6, linf

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

    # prepare model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_= tf.placeholder(tf.float32, [None, 11])

    model = build_graph_init(x, sess, output_units=11, pretrained=None)

    # define optimization process
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=y_)
    loss = tf.reduce_mean(loss)

    tvars = tf.trainable_variables()
    optimizer = tf.train.MomentumOptimizer(1e-2, 0.9, use_nesterov=True)
    optim_step = optimizer.minimize(loss=loss, var_list=tvars)
    sess.run(tf.variables_initializer(var_list=[var for var in tf.global_variables() if 'Momentum' in var.op.name]))
    print('Training variables:')
    for var in tvars:
        print(var.op.name)

    # define evaluation metrics
    accuracy = eval_acc(model.logits, y_)

    # start training
    x_test_merge = np.concatenate([x_test, x_adv[49000:]], axis=0)
    y_test_merge = np.concatenate([y_test, y_adv[49000:]], axis=0)
    for n_epochs in range(50):
        rand_ind = np.arange(49000, dtype=np.int32)
        np.random.shuffle(rand_ind)
        rand_ind = rand_ind[:5000]
        x_train_merge = np.concatenate([x_train, np.take(x_adv, rand_ind, axis=0)], axis=0)
        y_train_merge = np.concatenate([y_train, np.take(y_adv, rand_ind, axis=0)], axis=0)
        datagen.fit(x_train_merge)
        dataflow = datagen.flow(x_train_merge, y_train_merge, batch_size=128)

        for n_iters in range(50000 / 128):
            x_batch, y_batch = dataflow.next()
            _, loss_val, acc_val = sess.run([optim_step, loss, accuracy], feed_dict={x: x_batch, y_: y_batch})
            print('Epoch %d Step %d: loss=%f, accuracy=%f' % (n_epochs + 1, n_iters + 1, loss_val, acc_val))

        loss_val = acc_val = 0.
        for n_iters in range(11000 / 100):
            x_batch = x_test_merge[n_iters*100 : (n_iters+1)*100]
            y_batch = y_test_merge[n_iters*100 : (n_iters+1)*100]
            batch_loss_val, batch_acc_val = sess.run([loss, accuracy], feed_dict={x: x_batch, y_: y_batch})
            loss_val += batch_loss_val
            acc_val += batch_acc_val
        loss_val /= (11000 / 100)
        acc_val /= (11000 / 100)
        print('Epoch %d validation result: loss=%f, accuracy=%f' % (n_epochs + 1, loss_val, acc_val))

    saver = tf.train.Saver(var_list=tvars)
    saver.save(sess, '../saved_models/new_model/adv_model')

def adv_test():
    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
    x_adv, _ = load_adversarial_samples()
    x_adv_test = x_adv[49000:]
    y_adv_test = y_train[49000:]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_= tf.placeholder(tf.float32, [None, 10])

    model = build_graph_init(x, sess, output_units=11, pretrained='../saved_models/new_model/adv_model')

    accuracy = eval_acc(model.logits[:, :10], y_)
    adv_detect_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.logits, 1), 10), tf.float32), 0)

    # test on legitimate images
    validate(sess, accuracy, x, y_, x_test, y_test)

    # test on adversarial images
    acc_val = detect_rate_val = 0.
    num_imgs = len(x_adv_test)
    num_batches = num_imgs / 100
    for n_batch in range(num_batches):
        x_batch = x_adv_test[n_batch * 100 : (n_batch + 1) * 100]
        y_batch = y_adv_test[n_batch * 100 : (n_batch + 1) * 100]
        batch_detect_rate_val = sess.run(adv_detect_rate, feed_dict={x: x_batch})
        batch_acc_val = sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch})
        detect_rate_val += batch_detect_rate_val 
        acc_val += batch_acc_val
    detect_rate_val /= num_batches
    print('Detected rate on adversarial images: %f' % detect_rate_val)
    acc_val /= num_batches
    print('Test accurcacy on adversarial images: %f' % acc_val)

def adv_reverse():
    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()
    x_adv, _ = load_adversarial_samples()
    y_adv = y_train
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, [32, 32, 3])
    x_adv_ = tf.Variable(tf.zeros([32, 32, 3]))
    y_= tf.placeholder(tf.float32, [1, 10])

    assign_op = tf.assign(x_adv_, x)
    model = build_graph_init(x_adv_, sess, expand_dim=True, pretrained=PRETRAINED_PATH)
    accuracy = eval_acc(model.logits, y_)

    # initiate attack
    target = None
    targeted = (target is not None)
    eps_val = 0.2
    num_steps = 100

    adversary = Generator('FG', 32, x_adv_, model.logits, targeted=targeted)

    num_samples = 100

    reverse_rate = 0.
    reverse_imgs = []
    np.random.seed(2018)
    for i in range(num_samples):
        sample_x = x_adv[i]
        #sample_x += (np.random.rand(32, 32, 3) - 0.5) / 5 * 2
        sample_x = np.clip(np.clip(sample_x, x_adv[i] - eps_val, x_adv[i] + eps_val), 0., 1.)
        sample_y = y_adv[i]

        sess.run(assign_op, feed_dict={x: sample_x})
        pred = sess.run(tf.argmax(model.logits, axis=1))[0]
        if pred != np.argmax(sample_y):
            print('Image %d:' % i)
            reverse_img = adversary.generate(sess, x_adv[i], pred,
                eps_val=eps_val, num_steps=num_steps, loss_thresh=1.)#0.6931472)
            reverse_imgs.append(reverse_img)
            perturbation = reverse_img - x_train[i]
            evaluate_perturbation(perturbation)

            reverse_rate += sess.run(accuracy, feed_dict={y_: [sample_y]})
    print(reverse_rate / len(reverse_imgs))


if __name__ == '__main__':
    #adv_train()
    #adv_test()

    adv_reverse()

'''
target = None
targeted = (target is not None)
eps_val = 0.2
num_steps = 100

x = tf.placeholder(tf.float32, [32, 32, 3])
x_adv = tf.Variable(tf.zeros([32, 32, 3]))
y_= tf.placeholder(tf.float32, [1, 10])
assign_op = tf.assign(x_adv, x)

with tf.variable_scope('conv'):
    model = Classifier(x_adv)
adversary = Generator('FG', 32, x_adv, model.logits, targeted=targeted)

sess.run(tf.global_variables_initializer())

var_list = [var for var in tf.trainable_variables() if var.op.name.startswith('conv')]
saver = tf.train.Saver(var_list=var_list)
saver.restore(sess, PRETRAINED_PATH)
print('Restored variables:')
for var in var_list:
    print(var.op.name)

(x_train, y_train), (x_test, y_test) = data_loader.load_original_data()


correct = 0.

start_time = time.time()
for i in range(len(x_train)):
    sess.run(assign_op, feed_dict={x: x_train[i]})
    adv_img = adversary.generate(sess, x_train[i], np.argmax(y_train[i]), eps_val=eps_val, num_steps=num_steps)[-1]
    adv_img = np.around(np.array(adv_img) * 255)

    sess.run(assign_op, feed_dict={x: adv_img / 255.})
    correct += sess.run(accuracy, feed_dict={y_: [y_train[i]]})
    print('%d/%d: accuracy=%f' % (i + 1, len(x_train), correct / (i + 1)))

    plt.imsave('./adv_imgs/%d.png' % i, adv_img.astype(np.uint8))
print('Time spent: %ds' % (time.time() - start_time))
'''
