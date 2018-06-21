import os
import numpy as np
import tensorflow as tf
from model import Model
from utilities import Load_Cifar10_Data

TRAIN = 0
EVAL = 1
def validate_target_model(model):
    assert type(model) is Model
    (x_train, y_train), (x_test, y_test) = Load_Cifar10_Data()
    final_acc = model.test(x_test,y_test)
    print("accuracy on benign data: " + str(final_acc))
    
def filter_data(model,x_test,y_test,target = None, size = 128):
    num = x_test.shape[0]
    counter = 0
    assert counter <= num
    
    x = model.x
    y = model.y
    mode = model.mode
    sess = model.sess
    acc = model.acc
    
    
    x_selected = None
    y_selected = None
    for i in range(num):
        if counter >= size:
            break
        if sess.run(acc, feed_dict={x: x_test[i:i+1],y: y_test[i:i+1],mode: EVAL}) == 1 and np.argmax(y_test[i])!=target:
            if x_selected is None:
                x_selected = x_test[i:i+1]
            else:
                x_selected = np.concatenate((x_selected,x_test[i:i+1]),axis = 0)
            if y_selected is None:
                y_selected = y_test[i:i+1]
            else:
                y_selected = np.concatenate((y_selected,y_test[i:i+1]),axis = 0)
            counter += 1
    return x_selected,y_selected

def compute_success_rate(model,adv_imgs,y_test,target = None):
    x = model.x
    y = model.y
    mode = model.mode
    sess = model.sess
    acc = model.acc
    
    # evaluate the success rate:
    num = adv_imgs.shape[0]
    final_acc = 0
    if target is None:
        labels = y_test
    else:
        labels = np.repeat(np.eye(10)[target:target+1], num, axis = 0)
    for i in range(num):
            final_acc += sess.run(acc, feed_dict={x: adv_imgs[i:i+1],y: labels[i:i+1],mode: EVAL})
    final_acc /= num
    
    if target is None:
        final_acc = 1-final_acc
        print("success rate for non target attack is: " + str(final_acc))
    else:
        print("sccess rate for target attack on class " + str(target) + " is " + str(final_acc)) 
    return final_acc

def compute_distortion(origin,adv_imgs,norm = "linf"):
    assert norm in ["linf","l2"]
    if norm == "linf":
        return np.mean(np.amax(np.absolute(origin-adv_imgs),axis = (1,2,3)))
    elif norm == "l2":
        return np.mean(np.square(origin-adv_imgs))
    
def white_box_single_step(model,x_test, y_test, params):
    assert type(params) is dict
    for each in ["eps","target","clip_min","clip_max","batch_size","loss_type", "norm", "conf","alpha"]:
        assert each in params
        
    eps = params["eps"]
    target = params["target"] 
    clip_min = params["clip_min"]
    clip_max = params["clip_max"]
    batch_size = params["batch_size"]
    loss_type = params["loss_type"]
    norm = params["norm"]
    conf = params["conf"]
    alpha = params["alpha"]
    
    x = model.x
    y = model.y
    mode = model.mode
    logits = model.logits
    sess = model.sess
    acc = model.acc
    
    assert loss_type in ["xent","cw"]
    
    if loss_type == "xent":
        loss = model.loss
    elif loss_type == "cw":
        real = tf.reduce_sum(y*logits, 1)
        other = tf.reduce_max((1-y)*logits - (y*10000), 1)
        if target is None:
            loss = tf.maximum(0.0,real-other+conf)
        else:
            loss = tf.maximum(0.0,other-real+conf)
            
    grad = tf.gradients(loss,x)
    
    if norm == "linf":
        normed_grad = tf.sign(grad)
    elif norm == "l2":
        normed_grad = tf.nn.l2_normalize(grad, axis = (1,2,3))
        
    scaled_grad = (eps - alpha )* normed_grad
    
    if target is None:
        adv_x = tf.stop_gradient(x + scaled_grad)
        labels = y_test
    else:
        adv_x = tf.stop_gradient(x - scaled_grad)
        labels = np.repeat(np.eye(10)[target:target + 1],repeats = x_test.shape[0], axis = 0)
        
    if loss_type == "cw":
        adv_x = tf.stop_gradient(x-scaled_grad)
        
    adv_x = tf.clip_by_value(adv_x,clip_min,clip_max)
    
    # generate the adversarial samples:
    num = x_test.shape[0]//batch_size
    adv_imgs = None
    for i in range(num):
        adv_img = sess.run(adv_x,feed_dict = {x:x_test[i*batch_size:(i+1)*batch_size],y:labels[i*batch_size:(i+1)*batch_size],mode:EVAL})[0]
        print(adv_img.shape)
        if adv_imgs is None:
            adv_imgs = adv_img
        else:
            adv_imgs = np.concatenate((adv_imgs,adv_img), axis = 0)
            
    return adv_imgs

def white_box_iterative(model,x_test, y_test, params):
    assert type(params) is dict
    for each in ["eps", "eps_iter", "nb_iter", "target", "clip_min", "clip_max", "batch_size","loss_type", "norm", "conf","alpha"]:
        assert each in params
    eps = params["eps"]
    eps_iter = params["eps_iter"]
    nb_iter = params["nb_iter"]
    target = params["target"]
    clip_min = params["clip_min"]
    clip_max = params["clip_max"]
    batch_size = params["batch_size"]
    loss_type = params["loss_type"]
    norm = params["norm"]
    conf = params["conf"]
    alpha = params["alpha"]
    
    x = model.x
    y = model.y
    mode = model.mode
    logits = model.logits
    sess = model.sess
    acc = model.acc
    
    assert loss_type in ["xent","cw"]
    if loss_type == "xent":
        loss = model.loss
    elif loss_type == "cw":
        real = tf.reduce_sum(y*logits, 1)
        other = tf.reduce_max((1-y)*logits - (y*10000), 1)
        if target is None:
            loss = tf.maximum(0.0,real-other+conf)
        else:
            loss = tf.maximum(0.0,other-real+conf)
            
    grad = tf.gradients(loss,x)
    
    if norm == "linf":
        normed_grad = tf.sign(grad)
    elif norm == "l2":
        normed_grad = tf.nn.l2_normalize(grad, axis = (1,2,3))
        
    scaled_grad = eps_iter * normed_grad
    
    if target is None:
        adv_x = tf.stop_gradient(x + scaled_grad)
        labels = y_test
    else:
        adv_x = tf.stop_gradient(x - scaled_grad)
        labels = np.repeat(np.eye(10)[target:target + 1],repeats = x_test.shape[0], axis = 0)
    
    if loss_type == "cw":
        adv_x = tf.stop_gradient(x-scaled_grad)
        
    adv_x = tf.clip_by_value(adv_x,clip_min,clip_max)
    
    # generate the adversarial samples:
    num = x_test.shape[0]//batch_size
    adv_imgs = x_test
    for j in range(nb_iter):
        cur_adv_imgs = None
        diff = None
        for i in range(num):
            adv_img = sess.run(adv_x,feed_dict = {x:adv_imgs[i*batch_size:(i+1)*batch_size],y:labels[i*batch_size:(i+1)*batch_size],mode:EVAL})[0]
            batch_diff = adv_img - x_test[i*batch_size:(i+1)*batch_size]
            if cur_adv_imgs is None:
                cur_adv_imgs = adv_img
                diff = batch_diff
            else:
                cur_adv_imgs = np.concatenate((cur_adv_imgs,adv_img), axis = 0)
                diff = np.concatenate((diff,batch_diff), axis = 0) 
        adv_imgs = x_test[:num*batch_size] + np.clip(diff,-eps, eps)
        print((adv_imgs).shape)
    adv_imgs = np.clip(adv_imgs,clip_min,clip_max)
    return adv_imgs


def black_box_single_step(model, x_test, y_test, params):
    assert type(params) is dict
    for each in ["eps", "p", "target", "clip_min", "clip_max", "batch_size"]:
        assert each in params
    eps = params["eps"]
    p = params["p"]
    target = params["target"]
    clip_min = params["clip_min"]
    clip_max = params["clip_max"]
    batch_size = params["batch_size"]
    
    x = model.x
    y = model.y
    mode = model.mode
    logits = model.logits
    sess = model.sess
    acc = model.acc
    pred = model.pred
    
    from itertools import product
    _,L, M, N = x_test.shape
    
    labels = None
    if target is None:
        labels = y_test
    else:
        labels = np.repeat(np.eye(10)[target:target + 1],repeats = x_test.shape[0], axis = 0)
    # generate adversarial samples
    
    num = x_test.shape[0]//batch_size
    adv_imgs = None
    for i in range(num):
        x_batch = x_test[i*batch_size:(i+1)*batch_size]
        label_batch = labels[i*batch_size:(i+1)*batch_size]
        grad_batch = np.zeros_like(x_batch)
        for l,m,n in product(range(L), range(M), range(N)):
            
            pos = np.array(x_batch)
            pos[:,l,m,n] += p
            neg = np.array(x_batch)
            neg[:,l,m,n] -= p
            
            pos_prob = sess.run(pred,feed_dict = {x:pos,y:label_batch,mode:EVAL})
            neg_prob = sess.run(pred,feed_dict = {x:neg,y:label_batch,mode:EVAL})
            
            est_grad = - tf.reduce_sum(tf.multiply((pos_prob - neg_prob),y)) / (2 * p)
            est_grad = sess.run(est_grad,feed_dict={x:x_batch,y:label_batch,mode:EVAL})
            
            grad_batch[:,l,m,n] = est_grad
            print(l,m,n)
        signed_grad = np.sign(grad_batch)
        if target is None:
            adv_img = x_batch + eps*signed_grad
        else:
            adv_img = x_batch - eps*signed_grad
        if adv_imgs is None:
            adv_imgs = adv_img
        else:
            adv_imgs = np.concatenate((adv_imgs,adv_img), axis = 0)
        print(adv_imgs.shape)
    adv_imgs = np.clip(adv_imgs,clip_min,clip_max)
    
    return adv_imgs

def black_box_iterative(model,x_test, y_test, params):
    
    assert type(params) is dict
    for each in ["eps", "eps_iter", "p", "nb_iter", "target", "clip_min", "clip_max", "batch_size"]:
        assert each in params
    eps = params["eps"]
    eps_itr = params["eps_iter"]
    p = params["p"]
    nb_iter = params["nb_iter"]
    target = params["target"]
    clip_min = params["clip_min"]
    clip_max = params["clip_max"]
    batch_size = params["batch_size"]
    
    x = model.x
    y = model.y
    mode = model.mode
    sess = model.sess
    acc = model.acc
    
    from itertools import product
    _,L, M, N = x_test.shape
    
    labels = None
    if target is None:
        labels = y_test
    else:
        labels = np.repeat(np.eye(10)[target:target + 1],repeats = x_test.shape[0], axis = 0)
        
    # generate adversarial samples
    num = x_test.shape[0]//batch_size
    adv_imgs = x_test
    for j in range(nb_iter):
        cur_adv_imgs = None
        for i in range(num):
            original_x_batch = x_test[i*batch_size:(i+1)*batch_size]
            x_batch = adv_imgs[i*batch_size:(i+1)*batch_size]
            label_batch = labels[i*batch_size:(i+1)*batch_size]
            grad_batch = np.zeros_like(x_batch)
            for l,m,n in product(range(L), range(M), range(N)):

                pos = np.array(x_batch)
                pos[:,l,m,n] += p
                neg = np.array(x_batch)
                neg[:,l,m,n] -= p

                pos_prob = sess.run(pred,feed_dict = {x:pos,y:label_batch,mode:EVAL})
                neg_prob = sess.run(pred,feed_dict = {x:neg,y:label_batch,mode:EVAL})

                est_grad = - tf.reduce_sum(tf.multiply((pos_prob - neg_prob),y))
                est_grad = sess.run(est_grad,feed_dict={x:x_batch,y:label_batch,mode:EVAL})
                
                grad_batch[:,l,m,n] = est_grad
                print(l,m,n)
            signed_grad = np.sign(grad_batch)
            if target is None:
                adv_img = x_batch + eps_iter*signed_grad
            else:
                adv_img = x_batch - eps_iter*signed_grad
                
            adv_img = np.clip(adv_img,clip_min,clip_max)
            
            adv_img = original_x_batch + np.clip(adv_img - original_x_batch, -eps, eps)
            
            if cur_adv_imgs is None:
                cur_adv_imgs = adv_img
            else:
                cur_adv_imgs = np.concatenate((cur_adv_imgs,adv_img), axis = 0)   
        adv_imgs = cur_adv_imgs
        print(adv_imgs.shape)
    adv_imgs = np.clip(adv_imgs,clip_min,clip_max)
    return adv_imgs
    