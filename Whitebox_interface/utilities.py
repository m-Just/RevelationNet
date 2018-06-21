import os
import numpy as np
import tensorflow as tf

TRAIN = 0
EVAL = 1

def Load_Cifar10_Data(scaled = True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    if scaled:
        x_train = x_train/255
        x_test  = x_test/255

    y_train = y_train.reshape([y_train.shape[0]])
    y_test = y_test.reshape([y_test.shape[0]])

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    return (x_train, y_train), (x_test, y_test)

def Conv2d(inputs,filters,kernel_size,padding,strides,activation,name,noise_info):
    # In each of the info
    # type: "add" or "mul" or "none"
    # func: the tensorflow random function
    ############################################
    # Optional parameters for the noise_func:
    # mean
    # stddev
    # minval
    # maxval
    # seed
    assert type(noise_info) is dict
    assert "kernel" in noise_info
    assert "bias" in noise_info
    k_info = noise_info["kernel"]
    b_info = noise_info["bias"]
    assert "type" in k_info
    assert "type" in b_info
    
    # declare variables for the layer
    kernel_shape = [kernel_size,kernel_size,inputs.get_shape().as_list()[-1],filters]
    bias_shape = [filters]
    init = tf.random_normal(shape = kernel_shape)
    init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
    kernel = tf.Variable(init, dtype = tf.float32, name = name + "/kernel")
    bias = tf.Variable(tf.zeros(shape = bias_shape), dtype = tf.float32, name = name+"/bias")
    
    if k_info["type"] != "none":
        kernel_noise = k_info["func"](shape = kernel_shape, ** k_info["params"])
        if k_info["type"] == "add":
            kernel = tf.add(kernel,kernel_noise)
        elif k_info["type"] == "mul":
            kernel = tf.multiply(kernel, kernel_noise)
    if b_info["type"] != "none":
        bias_noise = b_info["func"](shape = bias_shape, ** b_info["params"])
        if b_info["type"] == "add":
            bias = tf.add(bias,bias_noise)
        elif b_info["type"] == "mul":
            bias = tf.multiply(bias,bias_noise)
    
    output = activation(tf.nn.bias_add(tf.nn.conv2d(input = inputs ,filter = kernel,strides = [1, strides, strides, 1], padding = padding, name = name),bias))
    print("out shape:", output.get_shape())
    return output
    
def Dense(inputs,units,activation,name, noise_info):
    # In each of the info
    # type: "add" or "mul" or "none"
    # func: the tensorflow random function
    ############################################
    # Optional parameters for the noise_func:
    # mean
    # stddev
    # minval
    # maxval
    # seed
    
    assert type(noise_info) is dict
    assert "kernel" in noise_info
    assert "bias" in noise_info
    k_info = noise_info["kernel"]
    b_info = noise_info["bias"]
    assert "type" in k_info
    assert "type" in b_info
    # declare the variables for the layer
    kernel_shape = inputs.get_shape().as_list()[1:] + [units]
    bias_shape = [units]
    init = tf.random_normal(shape = kernel_shape)
    init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
    kernel = tf.Variable(init, dtype = tf.float32, name = name +"/kernel")
    bias = tf.Variable(tf.zeros(shape = bias_shape), dtype = tf.float32, name = name + "/bias")
    
    if k_info["type"] != "none":
        kernel_noise = k_info["func"](shape = kernel_shape, ** k_info["params"])
        if k_info["type"] == "add":
            kernel = tf.add(kernel,kernel_noise)
        elif k_info["type"] == "mul":
            kernel = tf.multiply(kernel, kernel_noise)
    if b_info["type"] != "none":
        bias_noise = b_info["func"](shape = bias_shape, ** b_info["params"])
        if b_info["type"] == "add":
            bias = tf.add(bias,bias_noise)
        elif b_info["type"] == "mul":
            bias = tf.multiply(bias,bias_noise)
            
    output = activation(tf.nn.bias_add(tf.matmul(inputs,kernel),bias))      
    print("out shape:", output.get_shape())
    
    return output


def Target(x,y,mode,noise_infos):
    # declare target network
    with tf.variable_scope("conv") as scope:
        prev_layer = x
        prev_layer = Conv2d(
            inputs = prev_layer,
            filters = 64,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation =tf.nn.relu,
            name = 'conv2d_1',
            noise_info = noise_infos[0]
        )
        prev_layer = Conv2d(
            inputs = prev_layer,
            filters = 64,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation =tf.nn.relu,
            name = 'conv2d_2',
            noise_info = noise_infos[1]
        )
        prev_layer = tf.layers.max_pooling2d(
            inputs = prev_layer,
            pool_size = 2,
            strides = 2
        )
        prev_layer = Conv2d(
            inputs = prev_layer,
            filters = 128,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation =tf.nn.relu,
            name = 'conv2d_3',
            noise_info = noise_infos[2]
        )
        prev_layer = Conv2d(
            inputs = prev_layer,
            filters = 128,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation =tf.nn.relu,
            name = 'conv2d_4',
            noise_info = noise_infos[3]
        )
        prev_layer = tf.layers.max_pooling2d(
            inputs = prev_layer,
            pool_size = 2,
            strides = 2
        )

        prev_layer = tf.contrib.layers.flatten(prev_layer)

        prev_layer = Dense(
            inputs = prev_layer,
            units = 256,
            activation = tf.nn.relu,
            name = 'dense_1',
            noise_info = noise_infos[4]
        )
        
        prev_layer = tf.layers.dropout(prev_layer,training = mode == TRAIN)
        
        prev_layer = Dense(
            inputs = prev_layer,
            units = 256,
            activation = tf.nn.relu,
            name = 'dense_2',
            noise_info = noise_infos[5]
        )
        logits = Dense(
            inputs = prev_layer,
            units = 10,
            activation = lambda t:t,
            name = 'dense_3',
            noise_info = noise_infos[6]
        )
        
    pred = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = logits))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(y,1)),tf.float32),0)
    
    return logits,pred,loss,acc