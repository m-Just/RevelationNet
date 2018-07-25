import tensorflow as tf

class Classifier(object):
    def __init__(self, x, regularizer=None, expand_dim=True,
        output_units=10,
        conv_layer=tf.layers.conv2d,
        dense_layer=tf.layers.dense):

        self.regularizer = regularizer
        self.output_units = output_units
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
            kernel_regularizer = self.regularizer,
            name = 'conv2d_1',
        )
        prev_layer = self.conv_layer(
            inputs = prev_layer,
            filters = 64,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation = tf.nn.relu,
            kernel_regularizer = self.regularizer,
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
            kernel_regularizer = self.regularizer,
            name = 'conv2d_3',
        )
        prev_layer = self.conv_layer(
            inputs = prev_layer,
            filters = 128,
            kernel_size = 3,
            padding = 'VALID',
            strides = 1,
            activation = tf.nn.relu,
            kernel_regularizer = self.regularizer,
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
            kernel_regularizer = self.regularizer,
            name = 'dense_1',
        )
        prev_layer = self.dense_layer(
            inputs = prev_layer,
            units = 256,
            activation = tf.nn.relu,
            kernel_regularizer = self.regularizer,
            name = 'dense_2',
        )
        self.logits = self.dense_layer(
            inputs = prev_layer,
            units = self.output_units,
            activation = lambda t:t,
            kernel_regularizer = self.regularizer,
            name = 'dense_3',
        )

class NoisyClassifier(Classifier):
    def __init__(self, x, expand_dim=True,
        noise_on_conv=None,
        noise_on_dense='uniform',
        noise_on_kernel=True,
        noise_on_bias=True,
        noise_minval=0,
        noise_maxval=1,
        noise_stddev=1):

        self.noise_on_conv = noise_on_conv
        self.noise_on_dense = noise_on_dense
        self.noise_on_kernel = noise_on_kernel
        self.noise_on_bias = noise_on_bias
        self.noise_minval = noise_minval
        self.noise_maxval = noise_maxval
        self.noise_stddev = noise_stddev
        
        super(NoisyClassifier, self).__init__(x, None, expand_dim, self.noisy_conv2d, self.noisy_dense)

    def noisy_conv2d(self, inputs, filters, kernel_size, padding, strides, activation, kernel_regularizer, name):
        in_filters = inputs.get_shape().as_list()[-1]
        kernel_shape = [kernel_size, kernel_size, in_filters, filters]
        bias_shape = [filters]

        kernel = tf.get_variable(name + '/kernel',
            shape = kernel_shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(),
            regularizer=kernel_regularizer)
        bias = tf.get_variable(name + '/bias',
            shape=bias_shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
            
        if self.noise_on_conv is not None:
            assert self.noise_maxval >= self.noise_minval
            if self.noise_on_conv == 'normal':
                noise_mean = (self.noise_maxval + self.noise_minval) / 2.
                noise_diff = (self.noise_maxval - self.noise_minval) / 2.
                if self.noise_on_kernel:
                    kernel_noise = tf.truncated_normal(shape=kernel_shape, stddev=self.noise_stddev)
                    kernel_noise = kernel_noise / (2 * self.noise_stddev) * noise_diff + noise_mean
                    kernel = tf.multiply(kernel_noise, kernel)
                if self.noise_on_bias:
                    bias_noise = tf.truncated_normal(shape=bias_shape, stddev=self.noise_stddev)
                    bias_noise = bias_noise / (2 * self.noise_stddev) * noise_diff + noise_mean
                    bias = tf.multiply(bias_noise, bias)
            elif self.noise_on_conv == 'uniform':
                if self.noise_on_kernel:
                    kernel_noise = tf.random_uniform(shape=kernel_shape, minval=self.noise_minval, maxval=self.noise_maxval)
                    kernel = tf.multiply(kernel_noise, kernel)
                if self.noise_on_bias:
                    bias_noise = tf.random_uniform(shape=bias_shape, minval=self.noise_minval, maxval=self.noise_maxval)
                    bias = tf.multiply(bias_noise, bias)

        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding, name=name)
        return activation(tf.nn.bias_add(conv, bias))

    def noisy_dense(self, inputs, units, activation, kernel_regularizer, name):
        kernel_shape = inputs.get_shape().as_list()[1:] + [units]
        bias_shape = [units]

        kernel = tf.get_variable(name + '/kernel',
            shape=kernel_shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(),
            regularizer=kernel_regularizer)
        bias = tf.get_variable(name + '/bias',
            shape=bias_shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        if self.noise_on_dense is not None:
            assert self.noise_maxval >= self.noise_minval
            if self.noise_on_dense == 'normal':
                noise_mean = (self.noise_maxval + self.noise_minval) / 2.
                noise_diff = (self.noise_maxval - self.noise_minval) / 2.
                if self.noise_on_kernel:
                    kernel_noise = tf.truncated_normal(shape=kernel_shape, stddev=self.noise_stddev)
                    kernel_noise = kernel_noise / (2 * self.noise_stddev) * noise_diff + noise_mean
                    kernel = tf.multiply(kernel_noise, kernel)
                if self.noise_on_bias:
                    bias_noise = tf.truncated_normal(shape=bias_shape, stddev=self.noise_stddev)
                    bias_noise = bias_noise / (2 * self.noise_stddev) * noise_diff + noise_mean
                    bias = tf.multiply(bias_noise, bias)
            elif self.noise_on_dense == 'uniform':
                if self.noise_on_kernel:
                    kernel_noise = tf.random_uniform(shape=kernel_shape, minval=self.noise_minval, maxval=self.noise_maxval)
                    kernel = tf.multiply(kernel_noise, kernel)
                if self.noise_on_bias:
                    bias_noise = tf.random_uniform(shape=bias_shape, minval=self.noise_minval, maxval=self.noise_maxval)

        return activation(tf.nn.bias_add(tf.matmul(inputs, kernel), bias))

class InputPerturbedClassifier(Classifier):
    def __init__(self, x, minval, maxval):
        self.regularizer = None
        self.conv_layer = tf.layers.conv2d
        self.dense_layer = tf.layers.dense
        x = tf.expand_dims(x, 0)
        perturbed_x = x + tf.random_uniform(shape=x.get_shape(), minval=minval, maxval=maxval)
        super(InputPerturbedClassifier, self).build_graph(perturbed_x)
