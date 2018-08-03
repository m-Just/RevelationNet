from __future__ import print_function

import glob
import time
from six.moves import urllib

import numpy as np
import scipy
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

from classifiers import SimpleCNN, Resnet_v2_101
import data_loader
from gradient_attack import Generator, Tracker, Navigator, Backtracker

def build_graph_init(Classifier, x, sess, expand_dim=False, pretrained=None,
                     scope='conv'):

    model = Classifier(x, expand_dim=expand_dim)
    sess.run(tf.global_variables_initializer())

    if pretrained is not None:
        var_list = [var for var in tf.global_variables() if \
                    var.op.name.startswith(scope)]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, pretrained)
        
        print('Restored variables:')
        for var in var_list: print(var.op.name)
    
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

def evaluate_perturbation(p):
    abs_p = np.abs(p)
    l_one = np.sum(abs_p)
    l_two = np.sqrt(np.sum(p ** 2))
    l_inf = np.max(abs_p)
    return l_one, l_two, l_inf
    
def get_ranking(logits):
    ranking = [(label, logit) for label, logit in enumerate(logits)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    return [label for label, logit in ranking]

def direction_tracking(att_path, rec_path):

    normalize = lambda x: x / np.sqrt(np.sum(x ** 2))

    def stepwise(path, stepnum):
        cumulative = normalize(path[-1] - path[0])
        steps = [normalize(path[n] - path[n-1]) for n in range(1, stepnum)]
        cossim = [np.sum(cumulative * step) for step in steps]
        print('  Maximum=%f' % np.max(cossim))
        print('  Minimum=%f' % np.min(cossim))
        print('  Average=%f' % np.mean(cossim))
        return cumulative, steps

    att_stepnum = len(att_path)
    rec_stepnum = len(rec_path)
    print('Attack steps cosine similarity:')
    att_cm, att_steps = stepwise(att_path, att_stepnum)
    print('Recover steps cosine similarity:')
    rec_cm, rec_steps = stepwise(rec_path, rec_stepnum)

    print('Backtracking directional accuracy:')
    print('  First step=%f' % np.sum(att_steps[-1] * (-rec_steps[0])))
    print('  Last  step=%f' % np.sum(att_steps[0] * (-rec_steps[-1])))
    print('  Cumulative=%f' % np.sum(att_cm * (-rec_cm)))

def backtrack(attack_method='FG', target=None, eps_val=0.02, recover_eps_val=0.02,
              num_steps=100, noise_scale=1., sigma=0.5, recover_num_steps=100,
              attack_optimize_method='sgd', recover_optimize_method='sgd',
              recover_step_scale=1., recover_loss_thresh=None,
              use_experiment=None):

    eps_val = eps_val * (IMAGE_MAX - IMAGE_MIN)
    recover_eps_val = recover_eps_val * (IMAGE_MAX - IMAGE_MIN)

    # initiate attack
    targeted = (target is not None)
    adversary1 = Generator(attack_method, IMGSIZE, IMAGE_MIN, IMAGE_MAX,
                           x_adv_, model.logits, targeted=targeted,
                           optimize_method=attack_optimize_method,
                           cls_no=NUM_CLASSES)
    if use_experiment == 'tracker':
        adversary2 = Tracker(IMGSIZE, IMAGE_MIN, IMAGE_MAX, x_adv_, model.logits,
                             num_classes=NUM_CLASSES)
    elif use_experiment == 'navigator':
        adversary2 = Navigator(IMGSIZE, IMAGE_MIN, IMAGE_MAX, x_adv_,
                               model.logits, num_classes=NUM_CLASSES)
    elif use_experiment == 'backtracker':
        adversary2 = Backtracker(IMGSIZE, IMAGE_MIN, IMAGE_MAX, x_adv_,
                                 model.logits, num_classes=NUM_CLASSES)
    else:
        adversary2 = Generator('FG', IMGSIZE, IMAGE_MIN, IMAGE_MAX, x_adv_,
                               model.logits, targeted=False,
                               optimize_method=recover_optimize_method,
                               cls_no=NUM_CLASSES)

    noise_sampling_n = 10
    avg_l1 = avg_l2 = avg_linf = 0.
    recovered = 0.
    gt_rank_adv = [0. for i in range(NUM_CLASSES)]
    gt_rank_noisy = [0. for i in range(NUM_CLASSES)]
    gt_rank_rc = [0. for i in range(NUM_CLASSES)]
    result_imgs = []

    for i in range(num_samples):
        #if i not in [26, 49, 57, 63, 85, 91]: continue
        sess.run(assign_op, feed_dict={x: x_test[i]})
        lgt_logits, lgt_pred = sess.run([model.logits, prediction])
        print('\nLegitimate image %d' % i)
        print('  Ground-truth label: %d' % np.argmax(y_test[i]))
        print('  Predicted class:    %d' % lgt_pred)
        print('  Class rankings (legitimate):', get_ranking(lgt_logits[0])[:10])
        target_label = target if targeted else np.argmax(y_test[i])
        if targeted and lgt_pred == target_label:
            print('Ground truth label is target: skip image %d' % i)
            continue
        if lgt_pred != np.argmax(y_test[i]):
            print('Wrong prediction: skip image %d' % i)
            continue

        # generate adversarial image from test set on fly
        print('Generating adversarial image %d' % i)
        adv_path = adversary1.generate(sess, x_test[i], target_label,
            eps_val=eps_val, num_steps=num_steps)
        x_adv = adv_path[-1]
        y_adv = y_test[i]
        sess.run(assign_op, feed_dict={x: x_adv})
        logits, pred = sess.run([model.logits, prediction])
        if (not targeted and pred != np.argmax(y_adv)) or\
           (targeted and pred == target_label): # successful attack
            print('  Changed prediction %d -> %d' % (lgt_pred, pred))
        else:
            print('  Unsuccessful attack: skip image %d' % i)
            continue
        ranking = get_ranking(logits[0])
        gt_rank_adv[ranking.index(lgt_pred)] += 1
        print('  Class rankings (clean adversarial):', ranking[:10])
        l1, l2, linf = evaluate_perturbation(x_adv - x_test[i])
        print('  Perturbation L1  : %f' % l1)
        print('  Perturbation L2  : %f' % l2)
        print('  Perturbation Linf: %f' % linf)

        sample_y = y_adv
        if noise_scale == 0.:
            sample_x = x_adv
            sess.run(assign_op, feed_dict={x: sample_x})
        else: # apply adequate noise to the adversarial image
            print('Applying noise to adversarial image %d' % i)
            max_loss = -1.
            for n in range(noise_sampling_n):
                noise = np.random.rand(IMGSIZE, IMGSIZE, 3)
                noise = ((noise >= 0.5) - 0.5) * eps_val * 2 * noise_scale
                noisy_img = x_adv + noise
                sess.run(assign_op, feed_dict={x: noisy_img})
                loss = sess.run(adversary1.loss,
                                feed_dict={adversary1.y_adv: pred})
                if loss > max_loss:
                    max_loss = loss
                    sample_x = noisy_img
            sample_x = np.clip(sample_x, IMAGE_MIN, IMAGE_MAX)
            sess.run(assign_op, feed_dict={x: sample_x})

            #if pred != sess.run(prediction):
            #    print('Noise changed prediction: skip image %d' % i)
            #    continue
            logits = sess.run(model.logits)
            ranking = get_ranking(logits[0])
            gt_rank_noisy[ranking.index(lgt_pred)] += 1
            print('  Class rankings (noisy adversarial):', ranking[:10])

        # attack the adversarial image
        print('Recovering adversarial image %d' % i)
        if sigma is None:
            # using noisy one as base is much better than the clean one
            clipping_base = sample_x
        else:
            clipping_base = scipy.ndimage.filters.gaussian_filter(sample_x,
                sigma=sigma)

        if use_experiment == 'tracker':
            result_path = adversary2.generate(sess, x_adv,
                num_steps=int(recover_num_steps/recover_step_scale),
                step_scale=recover_step_scale)
        elif use_experiment == 'navigator':
            result_path = adversary2.generate(sess, x_adv, pred, 0.001, 10,
                num_steps=int(recover_num_steps/recover_step_scale),
                step_scale=recover_step_scale)
        elif use_experiment == 'backtracker':
            result_path = adversary2.generate(sess, x_adv, pred,
                eps_val=recover_eps_val,
                num_steps=int(recover_num_steps/recover_step_scale),
                step_scale=recover_step_scale)
        else:
            result_path = adversary2.generate(sess, clipping_base, pred,
                eps_val=recover_eps_val,
                num_steps=int(recover_num_steps/recover_step_scale),
                step_scale=recover_step_scale,
                loss_thresh=recover_loss_thresh)
        result_img = result_path[-1]
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
            '''
            n_count = [0 for _ in range(NUM_CLASSES)]
            for k in range(100):
                noise = np.random.rand(IMGSIZE, IMGSIZE, 3)
                noise = (noise >= 0.5) * 2. - 1.
                n_img = result_img + noise * eps_val * 2
                sess.run(assign_op, feed_dict={x: n_img})
                n_pred = sess.run(prediction)
                n_count[n_pred] += 1
                
            n_max = max(n_count)
            for cls, cnt in enumerate(n_count):
                if n_max == cnt: s_pred = cls
            print('  ' + str(n_count))
            print(lgt_logits[0])
            sess.run(assign_op, feed_dict={x: (result_img + x_adv) / 2.})
            s_logits, s_pred = sess.run([model.logits, prediction])
            print('  ' + str(get_ranking(s_logits[0])))
            print('  Suggested class:    %d' % s_pred)
            '''

        ranking = get_ranking(r_logits[0])
        gt_rank_rc[ranking.index(lgt_pred)] += 1
        print('  Ground-truth label: %d' % np.argmax(sample_y))
        print('  Predicted class:    %d' % r_pred)
        print('  Class rankings (after attempted recovery):', ranking[:10])
        l1, l2, linf = evaluate_perturbation(result_img - x_test[i])
        print('  Perturbation (recovered - legitimate)')
        print('    L1  : %f' % l1)
        print('    L2  : %f' % l2)
        print('    Linf: %f' % linf)
        l1, l2, linf = evaluate_perturbation(result_img - x_adv)
        print('  Perturbation (recovered - adversarial)')
        print('    L1  : %f' % l1)
        print('    L2  : %f' % l2)
        print('    Linf: %f' % linf)
        avg_l1 += l1
        avg_l2 += l2
        avg_linf += linf

        adv_path = np.insert(adv_path, 0, x_test[i], axis=0)
        result_path = np.insert(result_path, 0, sample_x, axis=0)
        direction_tracking(adv_path, result_path)
        #raw_input()

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
    if noise_scale > 0.:
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
        target=0,
        noise_scale=0.,
        sigma=None,
        attack_optimize_method='sgd',
        recover_optimize_method='sgd',
        recover_step_scale=1.,
        #recover_eps_val=0.04,
        use_experiment=None)#'backtracker')

    '''
    parameter_dict = {
        'attack_method': ['FG'],
        'target': [0],
        'eps_val': [0.01, 0.02, 0.03, 0.04],# 0.08, 0.14, 0.20],
        'recover_eps_val': [0.01, 0.02, 0.04, 0.08],
        'noise_scale': [0.],# 0.25, 0.5, 1.0, 2.0, 4.0],
        'sigma': [None],# 0.25, 0.5, 0.75, 1.0],
        'attack_optimize_method': ['sgd'],# 'momentum', 'adam'],
        'recover_optimize_method': ['sgd'],# 'momentum', 'adam'],
        'recover_step_scale': [1.],# 10., 100.],
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

    #target=0, single step, fgm clean
    #  failed(26, 49, 57, 63, 85, 91) 74 0.918919
    #target=0, single step, navigator(0.001, 10):
    #  failed(42, 49, 57, 63, 65, 91) 74 0.918919
    #target=0, single step, navigator(0.001, 100):
    #  failed(42, 57, 63, 85, 91) 74 0.932432
    #target=0, single step, navigator(0.001, 500):
    #  failed(42, 49, 57, 63, 85, 91) 74 0.918919
    #target=0, single step, navigator(0.01, 10):
    #  failed(i25, 33, 46, 53, 57, 63, 85, 87, 91) 74 0.878378
    #target=0, single step, navigator(0.01, 100):
    #  failed(17, 25, 26, 33, 46, 53, 57, 63, 68, 71, 85, 91) 74 0.824324
