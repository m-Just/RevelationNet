from __future__ import print_function

import glob
import time
from six.moves import urllib

import numpy as np
import scipy
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from skimage.transform import resize

from classifiers import SimpleCNN, Resnet_v2_101
import data_loader
from gradient_attack import Generator

def build_graph_init(Classifier, x, sess, expand_dim=False, pretrained=None, scope='conv'):

    model = Classifier(x, expand_dim=expand_dim)
    sess.run(tf.global_variables_initializer())

    if pretrained is not None:
        var_list = [var for var in tf.global_variables() if var.op.name.startswith(scope)]
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
    abs_p = np.abs(p)
    l_one = np.sum(abs_p)
    l_two = np.sqrt(np.sum(p ** 2))
    l_inf = np.max(abs_p)
    return l_one, l_two, l_inf
    
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

def get_ranking(sess, logits):
    ranking = [(label, logit) for label, logit in enumerate(logits)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    return [label for label, logit in ranking]

def backtrack(attack_method='FG', target=None, eps_val=0.02, num_steps=100,
              noise_scale=1., sigma=0.5, recover_num_steps=100,
              attack_optimize_method='sgd', recover_optimize_method='sgd',
              recover_step_scale=1., recover_loss_thresh=None):

    eps_val = eps_val * (IMAGE_MAX - IMAGE_MIN)

    # initiate attack
    targeted = (target is not None)
    adversary1 = Generator(attack_method, IMGSIZE, IMAGE_MIN, IMAGE_MAX, x_adv_, model.logits, targeted=targeted, optimize_method=attack_optimize_method, cls_no=NUM_CLASSES)
    adversary2 = Generator('FG', IMGSIZE, IMAGE_MIN, IMAGE_MAX, x_adv_, model.logits, targeted=False, optimize_method=recover_optimize_method, cls_no=NUM_CLASSES)

    noise_sampling_n = 10
    avg_l1 = avg_l2 = avg_linf = 0.
    recovered = 0.
    gt_rank_adv = [0. for i in range(NUM_CLASSES)]
    gt_rank_noisy = [0. for i in range(NUM_CLASSES)]
    gt_rank_rc = [0. for i in range(NUM_CLASSES)]
    result_imgs = []

    for i in range(num_samples):
        sess.run(assign_op, feed_dict={x: x_test[i]})
        logits, lgt_pred = sess.run([model.logits, prediction])
        print('\nLegitimate image %d' % i)
        print('  Ground-truth label: %d' % np.argmax(y_test[i]))
        print('  Predicted class:    %d' % lgt_pred)
        print('  Class rankings (legitimate):', get_ranking(sess, logits[0])[:10])
        target_label = target if targeted else np.argmax(y_test[i])
        if targeted and lgt_pred == target_label:
            print('Ground truth label is target: skip image %d' % i)
            continue
        if lgt_pred != np.argmax(y_test[i]):
            print('Wrong prediction: skip image %d' % i)
            continue

        # generate adversarial image from test set on fly
        print('Generating adversarial image %d' % i)
        x_adv = adversary1.generate(sess, x_test[i], target_label,
            eps_val=eps_val, num_steps=num_steps)[-1]
        y_adv = y_test[i]
        sess.run(assign_op, feed_dict={x: x_adv})
        logits, pred = sess.run([model.logits, prediction])
        if (not targeted and pred != np.argmax(y_adv)) or\
           (targeted and pred == target_label): # successful attack
            print('  Changed prediction %d -> %d' % (lgt_pred, pred))
        else:
            print('  Unsuccessful attack: skip image %d' % i)
            continue
        ranking = get_ranking(sess, logits[0])
        gt_rank_adv[ranking.index(lgt_pred)] += 1
        print('  Class rankings (clean adversarial):', ranking[:10])
        l1, l2, linf = evaluate_perturbation(x_adv - x_test[i])
        print('  Perturbation L1  : %f' % l1)
        print('  Perturbation L2  : %f' % l2)
        print('  Perturbation Linf: %f' % linf)

        # apply adequate noise to the adversarial image
        print('Applying noise to adversarial image %d' % i)
        max_loss = -1.
        for n in range(noise_sampling_n): # search for random noise that maximize the loss
            #noise = (np.random.rand(32, 32, 3) - 0.5) * eps_val * 2
            noise = np.random.rand(IMGSIZE, IMGSIZE, 3)
            noise = ((noise >= 0.5) - 0.5) * eps_val * 2 * noise_scale
            noisy_img = x_adv + noise
            sess.run(assign_op, feed_dict={x: noisy_img})
            loss = sess.run(adversary1.loss, feed_dict={adversary1.y_adv: pred})
            if loss > max_loss:
                max_loss = loss
                sample_x = noisy_img
        sample_x = np.clip(sample_x, IMAGE_MIN, IMAGE_MAX)
        sample_y = y_adv

        sess.run(assign_op, feed_dict={x: sample_x})
        #if pred != sess.run(prediction): # noise should not change prediction
        #    print('Noise changed prediction: skip image %d' % i)
        #    continue
        logits = sess.run(model.logits)
        ranking = get_ranking(sess, logits[0])
        gt_rank_noisy[ranking.index(lgt_pred)] += 1
        print('  Class rankings (noisy adversarial):', ranking[:10])

        # attack the adversarial image
        print('Recovering adversarial image %d' % i)
        if sigma is None:
            # using noisy one as base is much better than the clean one
            clipping_base = sample_x
        else:
            clipping_base = scipy.ndimage.filters.gaussian_filter(sample_x, sigma=sigma)
        result_img = adversary2.generate(sess, clipping_base, pred,
            eps_val=eps_val,
            num_steps=int(recover_num_steps/recover_step_scale),
            step_scale=recover_step_scale,
            loss_thresh=recover_loss_thresh)
        result_imgs.append(result_img)

        # evaluate recovery status
        r_logits, r_pred, r_correct = sess.run(
            [model.logits, prediction, accuracy],
            feed_dict={y_: [sample_y]})
        if r_correct == 1.:
            print('Image %d successfully recovered' % i)
            recovered += 1
        else:
            print('Image %d failed to recover' % i)
        ranking = get_ranking(sess, r_logits[0])
        gt_rank_rc[ranking.index(lgt_pred)] += 1
        print('  Ground-truth label: %d' % np.argmax(sample_y))
        print('  Predicted class:    %d' % r_pred)
        print('  Class rankings (after attempted recovery):', ranking[:10])
        l1, l2, linf = evaluate_perturbation(result_img - x_test[i])
        print('  Perturbation L1  : %f' % l1)
        print('  Perturbation L2  : %f' % l2)
        print('  Perturbation Linf: %f' % linf)
        avg_l1 += l1
        avg_l2 += l2
        avg_linf += linf

    recovery_rate = recovered / len(result_imgs)
    avg_l1 /= len(result_imgs)
    avg_l2 /= len(result_imgs)
    avg_linf /= len(result_imgs)

    print()
    print('Recovery attempts: %d/%d' % (len(result_imgs), num_samples))
    print('Recovery rate: %f' % recovery_rate)

    print('Ranking percentage of ground truth label (clean adversarial)')
    for i in range(10):
        print('  #%d: %f' % (i + 1, gt_rank_adv[i] / len(result_imgs)))
    print('Ranking percentage of ground truth label (noisy adversarial)')
    for i in range(10):
        print('  #%d: %f' % (i + 1, gt_rank_noisy[i] / len(result_imgs)))
    print('Ranking percentage of ground truth label (after attempted recovery)')
    for i in range(10):
        print('  #%d: %f' % (i + 1, gt_rank_rc[i] / len(result_imgs)))

    return recovery_rate, avg_l1, avg_l2, avg_linf

if __name__ == '__main__':

    dataset = 'cifar10'
    model_name = 'simple'
    #dataset = 'imagenet'
    #model_name = 'resnet_v2_101'

    num_samples = 100
    np.random.seed(2018)

    if dataset == 'cifar10':
        IMAGE_MIN = 0
        IMAGE_MAX = 1
        NUM_CLASSES = 10
        IMGSIZE = 32
        (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()

    elif dataset == 'imagenet':

        IMAGE_MIN = -1
        IMAGE_MAX = 1
        NUM_CLASSES = 1001

        if model_name.startswith('resnet_v2'):
            IMGSIZE = 299
        else:
            IMGSIZE = 224

        x_test = []
        y_test = []

        synset_url = 'https://raw.githubusercontent.com/tensorflow/models/' + \
        'master/research/inception/inception/data/imagenet_lsvrc_2015_synsets.txt'
        filename, _ = urllib.request.urlretrieve(synset_url)
        synset_list = [s.strip() for s in open(filename).readlines()]
        num_synsets_in_ilsvrc = len(synset_list)
        assert num_synsets_in_ilsvrc == 1000

        image_paths = []
        labels = []
        for n in range(num_synsets_in_ilsvrc):
            path = './ImageNet/val/%s/*' % synset_list[n]
            temp = sorted(glob.glob(path))
            image_paths.extend(temp)
            labels.extend([n+1] * len(temp))
        ind = np.random.randint(len(image_paths), size=num_samples)
        for n in ind:
            image = plt.imread(image_paths[n])
            image = resize(image, [IMGSIZE, IMGSIZE]).astype(np.float32)
            image = image * 2. - 1.
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            assert image.shape == (IMGSIZE, IMGSIZE, 3)
            x_test.append(image)

            y_label = [0] * NUM_CLASSES
            y_label[labels[n]] = 1
            y_label = np.array(y_label, dtype=np.float32)
            y_test.append(y_label)

        assert len(x_test) == len(y_test)
            
        
    if model_name == 'simple':
        Classifier = SimpleCNN
        PRETRAINED_PATH = '../saved_models/pretrained_model'
        SCOPE = 'conv'
    elif model_name == 'resnet_v2_101':
        Classifier = Resnet_v2_101
        PRETRAINED_PATH = './imagenet_models/%s/%s.ckpt' %(model_name, model_name)
        SCOPE = model_name
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    x = tf.placeholder(tf.float32, [IMGSIZE, IMGSIZE, 3])
    x_adv_ = tf.Variable(tf.zeros([IMGSIZE, IMGSIZE, 3]))
    y_= tf.placeholder(tf.float32, [1, NUM_CLASSES])

    assign_op = tf.assign(x_adv_, x)
    model = build_graph_init(Classifier, x_adv_, sess, expand_dim=True, pretrained=PRETRAINED_PATH, scope=SCOPE)
    prediction = tf.squeeze(tf.argmax(model.logits, axis=1))
    accuracy = eval_acc(model.logits, y_)


    # non-targeted attack multistep recovery with noise and gaussian base clipping
    #backtrack(target=None, noise_scale=2, sigma=0.5)

    # targeted attack single step recovery
    #backtrack(target=1, noise_scale=0, sigma=None, recover_step_scale=100.)

    backtrack(
        attack_method='FG',
        target=None,
        noise_scale=0.,
        sigma=None,
        attack_optimize_method='sgd',
        recover_optimize_method='sgd',
        recover_step_scale=100.)

    '''
    parameter_dict = {
        'attack_method': ['FG'],
        'target': [None],
        'eps_val': [0.02, 0.08, 0.14, 0.20],
        'noise_scale': [2.],# 0.25, 0.5, 1.0, 2.0, 4.0],
        'sigma': [0.5],# 0.25, 0.5, 0.75, 1.0],
        'attack_optimize_method': ['sgd'],# 'momentum', 'adam'],
        'recover_optimize_method': ['sgd'],# 'momentum', 'adam'],
        'recover_step_scale': [1., 10., 100.],
    }
    from itertools import product
    f = open('./logs/backtrack_results.txt', 'w')
    for d in product(*[[(k, v) for v in l] for k, l in parameter_dict.items()]):
        rr, l1, l2, linf = backtrack(**dict(d))
        f.write('Parameters:\n')
        for k, v in d:
            f.write('  ' + k + '=' + str(v) + '\n')
        f.write('Results:\n')
        f.write('  Recovery rate=%f\n' % rr)
        f.write('  Average L1=%f\n' % l1)
        f.write('  Average L2=%f\n' % l2)
        f.write('  Average Linf=%f\n' % linf)
        f.write('\n')
        f.flush()
    f.close()
    '''
