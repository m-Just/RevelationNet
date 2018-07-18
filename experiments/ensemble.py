import os
import shutil
import random

from collections import OrderedDict

import tensorflow as tf
import numpy as np

def twin(BaseClassifier, batch_size=128):

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # build graph
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_= tf.placeholder(tf.float32, [None, 10])

    def build_and_init_model(modelname):
        with tf.variable_scope(modelname, reuse=tf.AUTO_REUSE):
            model = BaseClassifier(x, expand_dim=False)

            layers = OrderedDict()
            layers['conv1'] = tf.get_variable('conv2d_1/kernel')
            #layers['conv3'] = tf.get_variable('conv2d_3/kernel')

            cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=y_)
            cls_loss = tf.reduce_mean(cls_loss)

            correct_pred = tf.equal(tf.argmax(model.logits, 1), tf.argmax(y_, 1))
            cls_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=modelname)
        var_list = dict([('conv'+var.op.name[6:], var) for var in var_list])
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, PRETRAINED_PATH)

        return model, layers, cls_loss, cls_acc

    modelA, layersA, cls_lossA, cls_accA = build_and_init_model('modelA')
    modelB, layersB, cls_lossB, cls_accB = build_and_init_model('modelB')
    assert layersA.keys() == layersB.keys()

    tvars = []
    cossim = OrderedDict()
    for layer in layersA:
        normalizedA = tf.nn.l2_normalize(layersA[layer], axis=(0, 1))
        normalizedB = tf.nn.l2_normalize(layersB[layer], axis=(0, 1))
        cossim[layer] = tf.reduce_sum(normalizedA * normalizedB, axis=(0, 1))
        cossim[layer] = tf.reduce_mean(cossim[layer])
        
        tvars.append(layersA[layer])
        tvars.append(layersB[layer])

    print('Training variables:')
    for var in tvars:
        print(var.op.name)

    total_loss = cls_lossA + cls_lossB
    for layer in cossim:
        total_loss += 5 * (1. + cossim[layer])
        #total_loss += 10 * tf.maximum(cossim[layer], 0.)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.009808,
        momentum=0.9,
        use_nesterov=True)
    optim_step = optimizer.minimize(loss=total_loss, var_list=tvars)

    init_vars = [var for var in tf.global_variables() if 'Momentum' in var.op.name]
    sess.run(tf.variables_initializer(var_list=init_vars))

    # prepare data
    datagen, (x_train, y_train), (x_test, y_test) = data_loader.load_augmented_data()
    dataflow = datagen.flow(x_train, y_train, batch_size=batch_size)

    def evaluate():
        accA = accB = batch_num = 0.
        for image, label in zip(x_test, y_test):
            image = np.expand_dims(image, 0)
            label = np.expand_dims(label, 0)
            _accA, _accB = sess.run([cls_accA, cls_accB], feed_dict={x: image, y_: label})
            accA += _accA
            accB += _accB
            batch_num += 1
        accA /= batch_num
        accB /= batch_num
        print(accA)
        print(accB)

    for n_epoch in range(5):
        for n_batch in range(50000 / batch_size):
            image_batch, label_batch = dataflow.next()
            _, total_loss_val, cossim_vals, lossA, lossB, accA, accB = sess.run(
                [optim_step, total_loss, cossim, cls_lossA, cls_lossB, cls_accA, cls_accB],
                feed_dict={x: image_batch, y_: label_batch})

            if (n_batch + 1) % 10 == 0:
                print('Epoch %d, batch %d:' % (n_epoch+1, n_batch+1))
                print('  total_loss=%.6f' % total_loss_val)
                print('  ' + ' '.join([('cs_%s=%.6f' % item) for item in cossim_vals.items()]))
                print('  model-A: acc=%.4f, loss=%.6f' % (accA, lossA))
                print('  model-B: acc=%.4f, loss=%.6f' % (accB, lossB))

        evaluate()

    saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.op.name.startswith('modelA')])
    saver.save(sess, '../saved_models/new_model/ensemble_modelA')
    saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.op.name.startswith('modelB')])
    saver.save(sess, '../saved_models/new_model/ensemble_modelB')

def transfer_attack(BaseClassifier, target_label=None):
    x = tf.placeholder(tf.float32, [32, 32, 3])
    x_adv = tf.Variable(tf.zeros([32, 32, 3]))
    y_= tf.placeholder(tf.float32, [1, 10])

    assign_op = tf.assign(x_adv, x)
    with tf.variable_scope('modelA'):
        modelA = BaseClassifier(x_adv)
    with tf.variable_scope('modelB'):
        modelB = BaseClassifier(x)

    from FGSM import Generator
    targeted = (target_label is not None)
    fgsm_agent = Generator(32, x_adv, modelA.logits, targeted=targeted)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # original model with xavier kernel initialization
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='modelA')
    var_list = dict([('conv'+var.op.name[6:], var) for var in var_list])
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(sess, PRETRAINED_PATH)

    # original model with random normal kernel initialization
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='modelB')
    var_list = dict([('conv'+var.op.name[6:], var) for var in var_list])
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(sess, './saved_models/pretrained_model_rn')

    # ensemble models
    #saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.op.name.startswith('modelA')])
    #saver.restore(sess, '../saved_models/new_model/ensemble_modelA')
    #saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.op.name.startswith('modelB')])
    #saver.restore(sess, '../saved_models/new_model/ensemble_modelB')

    adv_imgs = []
    y_real = []
    (x_train, y_train), (x_test, y_test) = data_loader.load_original_data()

    indices = range(len(y_test))
    random.seed(2018)
    random.shuffle(indices)
    
    def eval_acc(logits, labels):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32), 0)

    # use model-A to generate adversarial images and test transferability on model-B
    eps_val = 0.2
    p_upper = eps_val + 1e-6
    p_lower = 0. - 1e-6
    num_steps = 100

    correctA = 0
    correctB = 0
    fidelityA = deceivedA = 0.
    fidelityB = deceivedB = 0.
    predA = [0. for _ in range(10)]
    predB = [0. for _ in range(10)]
    num_samples = 100
    for i in range(num_samples):
        sample_x = x_test[indices[i]]
        sample_y = y_test[indices[i]]

        sess.run(assign_op, feed_dict={x: sample_x})
        eval_accA = eval_acc(modelA.logits, y_)
        eval_accB = eval_acc(modelB.logits, y_)
        acc_valA = sess.run(eval_accA, feed_dict={x: sample_x, y_: [sample_y]})
        acc_valB = sess.run(eval_accB, feed_dict={x: sample_x, y_: [sample_y]})
        correctA += acc_valA
        correctB += acc_valB
        if (not targeted) or (acc_valA == 1 and acc_valB == 1 and np.argmax(sample_y) != target_label):
            if not targeted:
                target_label = np.argmax(sample_y)

            adv_img = fgsm_agent.generate(sess, sample_x, target_label,
                eps_val=eps_val, num_steps=num_steps)
            adv_imgs.append(adv_img[-1])
            perturbation = adv_img[-1] - sample_x
            y_real.append(sample_y)

            # select perturbations within a specific range to apply
            p_abs = np.abs(perturbation)
            adv_feed = np.where(np.all([p_abs <= p_upper, p_abs >= p_lower], axis=0), adv_img[-1], sample_x)
            p_feed = np.count_nonzero(adv_feed - sample_x) / float(32 * 32 * 3)
            print('Selected perturbation within [%.2f, %.2f]: %.2f' % (p_lower, p_upper, p_feed))
            sess.run(assign_op, feed_dict={x: adv_feed})
                
            # test accuracy
            fidelityA += sess.run(eval_accA, feed_dict={y_: [sample_y]})
            fidelityB += sess.run(eval_accB, feed_dict={x: adv_feed, y_: [sample_y]})
            if targeted:
                target_y = np.eye(10)[target_label]
                deceivedA += sess.run(eval_accA, feed_dict={y_: [target_y]})
                deceivedB += sess.run(eval_accB, feed_dict={x: adv_feed, y_: [target_y]})
            else:
                predA[sess.run(tf.squeeze(tf.argmax(modelA.logits, axis=1)))] += 1
                predB[sess.run(tf.squeeze(tf.argmax(modelB.logits, axis=1)), feed_dict={x: adv_feed})] += 1

                #print(sess.run(tf.nn.softmax(modelB.logits), feed_dict={x: sample_x}))
                #print(sess.run(tf.nn.softmax(modelB.logits), feed_dict={x: adv_img[-1]}))

            # perturbation analytical metrics
            d = 32 * 32 * 3

            print('L-one: %.4f' % sess.run(tf.norm(perturbation, ord=1)))
            print('L-two: %.4f' % sess.run(tf.norm(perturbation, ord=2)))
            print('L-inf: %.4f' % sess.run(tf.norm(perturbation, ord=np.inf)))
            RMSD = sess.run(tf.norm(perturbation, ord=2) / (d ** 0.5))
            print('RMSD : %.4f/1 | %.4f/255' % (RMSD, RMSD * 255))

            num_sectors = 11
            for p in range(num_sectors):
                upper_bound = eps_val * (num_sectors - 2 * p) / num_sectors + 1e-6
                lower_bound = eps_val * (num_sectors - 2 * (p + 1)) / num_sectors - 1e-6
                if upper_bound > 0 and lower_bound > 0:
                    a = np.where(np.where(perturbation <= upper_bound, perturbation, -1) > lower_bound, 1, 0)
                    c = np.count_nonzero(a)
                    print('Perturbation within (%+.2f, %+.2f]: %.2f' % (lower_bound, upper_bound, float(c) / d))
                elif upper_bound < 0 and lower_bound < 0:
                    a = np.where(np.where(perturbation < upper_bound, perturbation, -1) >= lower_bound, 1, 0)
                    c = np.count_nonzero(a)
                    print('Perturbation within [%+.2f, %+.2f): %.2f' % (lower_bound, upper_bound, float(c) / d))
                else:
                    a = np.where(np.where(perturbation <= upper_bound, perturbation, -1) >= lower_bound, 1, 0)
                    c = np.count_nonzero(a)
                    print('Perturbation within [%+.2f, %+.2f]: %.2f' % (lower_bound, upper_bound, float(c) / d))

    print(('Targeted' if targeted else 'Non-targeted') + ' transfer attack results:')
    print('Model-A Classification accuracy: %f' % (correctA / num_samples))
    print('Model-B Classification accuracy: %f' % (correctB / num_samples))
    print('Generated adversarial images %d/%d' % (len(adv_imgs), num_samples))

    n = len(adv_imgs)
    print('Model-A Fidelity rate on test set: %f' % (fidelityA / n))
    print('Model-B Fidelity rate on test set: %f' % (fidelityB / n))
    if targeted:
        print('Model-A Deceived rate on test set: %f' % (deceivedA / n))
        print('Model-B Deceived rate on test set: %f' % (deceivedB / n))
    else:
        for i in range(10):
            print('Model-A predictied as label-%d: %f' % (i, predA[i] / n))
        for i in range(10):
            print('Model-B predictied as label-%d: %f' % (i, predB[i] / n))

if __name__ == '__main__':
    from classifiers import Classifier
    import data_loader

    PRETRAINED_PATH = '../saved_models/pretrained_model'

    #twin(Classifier)
    transfer_attack(Classifier, target_label=None)
