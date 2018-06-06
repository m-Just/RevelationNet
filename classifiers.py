import tensorflow as tf

class Classifier(object):
    def __init__(self, x, expand_dim=True,
        conv_layer=tf.layers.conv2d,
        dense_layer=tf.layers.dense):

        self.conv_layer = conv_layer
        self.dense_layer = dense_layer

        if expand_dim:
            x = tf.expand_dims(x, 0)
        self.build_graph(x)
        
    def build_graph(self, x):
        prev_layer = self.conv_layer(
            inputs = x,
            filters = 64,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation = tf.nn.relu,
            name = 'conv2d_1',
        )
        prev_layer = self.conv_layer(
            inputs = prev_layer,
            filters = 64,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation = tf.nn.relu,
            name = 'conv2d_2',
        )
        prev_layer = tf.layers.max_pooling2d(
            inputs = prev_layer,
            pool_size = 2,
            strides = 2
        )

        prev_layer = self.conv_layer(
            inputs = prev_layer,
            filters = 128,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation = tf.nn.relu,
            name = 'conv2d_3',
        )
        prev_layer = self.conv_layer(
            inputs = prev_layer,
            filters = 128,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation = tf.nn.relu,
            name = 'conv2d_4',
        )
        prev_layer = tf.layers.max_pooling2d(
            inputs = prev_layer,
            pool_size = 2,
            strides = 2
        )

        prev_layer = tf.contrib.layers.flatten(prev_layer)
        flat = prev_layer

        prev_layer = self.dense_layer(
            inputs = prev_layer,
            units = 256,
            activation = tf.nn.relu,
            name = 'dense_1',
        )
        prev_layer = self.dense_layer(
            inputs = prev_layer,
            units = 256,
            activation = tf.nn.relu,
            name = 'dense_2',
        )
        self.logits = self.dense_layer(
            inputs = prev_layer,
            units = 10,
            activation = lambda t:t,
            name = 'dense_3',
        )

def NoisyClassifier(Classifier):
    def __init__(self, x, expand_dim=True,
        noise_on_conv=None,
        noise_on_dense='uniform',
        noise_minval=0,
        noise_maxval=1,
        noise_stddev=1):
        
        super(Classifier, self).__init__(x, expand_dim, self.noisy_conv2d, self.noisy_dense)

    def noisy_conv2d(self, inputs, filters, kernel_size, padding, strides, activation, name, noise_on_kernel=True, noise_on_bias=True):
        in_filters = inputs.get_shape().as_list()[-1]
        kernel_shape = [kernel_size, kernel_size, in_filters, filters]
        bias_shape = [filters]

        kernel = tf.get_variable(name + '/kernel',
            shape = kernel_shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable(name + '/bias',
            shape=bias_shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
            
        kernel_noise = tf.random_normal(shape=kernel_shape, stddev=0.5)
        bias_noise = tf.random_normal(shape=bias_shape, stddev=0.5)

        if noise_on_kernel:
            kernel = tf.multiply(1 + kernel_noise, kernel)
        if noise_on_bias:
            bias = tf.multiply(1 + bias_noise, bias)

        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding, name=name)
        return activation(tf.nn.bias_add(conv, bias))

    def noisy_dense(self, inputs, units, activation, name, noise_on_kernel=True, noise_on_bias=True):
        kernel_shape = inputs.get_shape().as_list()[1:] + [units]
        bias_shape = [units]

        kernel = tf.get_variable(name + '/kernel',
            shape=kernel_shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable(name + '/bias',
            shape=bias_shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        kernel_noise = tf.random_normal(shape=kernel_shape, stddev=0.5)
        bias_noise = tf.random_normal(shape=bias_shape, stddev=0.5)

        if noise_on_kernel:
            kernel = tf.multiply(1 + kernel_noise, kernel)
        if noise_on_bias:
            bias = tf.multiply(1 + bias_noise, bias)

        return activation(tf.nn.bias_add(tf.matmul(inputs, kernel), bias))
