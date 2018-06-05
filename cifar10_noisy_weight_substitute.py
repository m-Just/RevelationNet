
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
np.set_printoptions(threshold=np.nan)

TRAIN = 0
EVAL  = 1


# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255
x_test = x_test/255

y_train = y_train.reshape([y_train.shape[0]])
y_test = y_test.reshape([y_test.shape[0]])

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


# In[3]:


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

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)
flow = datagen.flow(x_train, y_train,batch_size=128)


# In[4]:


train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size = 1280).repeat()
train = train.batch(128)
train_itr = train.make_initializable_iterator()
next_batch = train_itr.get_next()


# In[5]:


# input
x = tf.Variable(tf.zeros([32,32,3]))
y_= tf.placeholder(tf.float32,[1,10])
# whether is training or not
mode = tf.placeholder(tf.int32,[])


# In[6]:


# cross_entropy   
def ce(y_pred, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels ,logits = y_pred))

# accuracy    
def acc(y_pred, labels):
    correct_prediction =tf.equal(tf.argmax(y_pred,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32),0)

def noisy_conv2d(inputs,filters,kernel_size,padding,strides,activation,name,scope_name = "conv", noise_on_kernel = True,noise_on_bias = True):
    print("in shape:", inputs.get_shape())
    kernel_shape = [kernel_size,kernel_size,inputs.get_shape().as_list()[-1],filters]
    print("kernel shape:", kernel_shape)
    bias_shape = [filters]
    kernel = tf.Variable(np.random.normal(size = kernel_shape), dtype = tf.float32, name = name + "/kernel")
    bias = tf.Variable(np.random.normal(size = bias_shape), dtype = tf.float32, name = name+"/bias")

    kernel_noise = tf.random_normal(shape = kernel_shape, stddev= 0)
    bias_noise = tf.random_normal(shape = bias_shape, stddev= 0)

    if noise_on_kernel:
        kernel = tf.multiply(1 + kernel_noise, kernel)
    if noise_on_bias:
        bias = tf.multiply(1 + bias_noise, bias)
    output = activation(tf.nn.bias_add(tf.nn.conv2d(input = inputs ,filter = kernel,strides = [1, strides, strides, 1], padding = padding, name = name),bias))
    print("out shape:", output.get_shape())
    return output
    
def noisy_dense(inputs,units,activation,name,scope_name = "conv", noise_on_kernel = True, noise_on_bias = True):
    kernel_shape = inputs.get_shape().as_list()[1:] + [units]
    bias_shape = [units]
    kernel = tf.Variable(np.random.normal(size = kernel_shape), dtype = tf.float32, name = name +"/kernel")
    bias = tf.Variable(np.random.normal(size = bias_shape), dtype = tf.float32, name = name + "/bias")

    kernel_noise = tf.random_normal(shape = kernel_shape, stddev= 0)
    bias_noise = tf.random_normal(shape = bias_shape, stddev= 0)

    if noise_on_kernel:
        kernel = tf.multiply(1 + kernel_noise, kernel)
    if noise_on_bias:
        bias = tf.multiply(1 + bias_noise, bias)
    return activation(tf.nn.bias_add(tf.matmul(inputs,kernel),bias))


# In[7]:


with tf.variable_scope("conv") as scope:
    prev_layer = tf.expand_dims(x, 0)
    prev_layer = noisy_conv2d(
        inputs = prev_layer,
        filters = 64,
        kernel_size = 3,
        padding = 'VALID',
        strides = 1,
        activation =tf.nn.relu,
        name = 'conv2d_1',
        scope_name = "conv"
    )
    prev_layer = noisy_conv2d(
        inputs = prev_layer,
        filters = 64,
        kernel_size = 3,
        padding = 'VALID',
        strides = 1,
        activation =tf.nn.relu,
        name = 'conv2d_2',
        scope_name = "conv"
    )
    prev_layer = tf.layers.max_pooling2d(
        inputs = prev_layer,
        pool_size = 2,
        strides = 2
    )


    prev_layer = noisy_conv2d(
        inputs = prev_layer,
        filters = 128,
        kernel_size = 3,
        padding = 'VALID',
        strides = 1,
        activation =tf.nn.relu,
        name = 'conv2d_3',
        scope_name = "conv"
    )
    prev_layer = noisy_conv2d(
        inputs = prev_layer,
        filters = 128,
        kernel_size = 3,
        padding = 'VALID',
        strides = 1,
        activation =tf.nn.relu,
        name = 'conv2d_4',
        scope_name = "conv"
    )
    prev_layer = tf.layers.max_pooling2d(
        inputs = prev_layer,
        pool_size = 2,
        strides = 2
    )

    prev_layer = tf.contrib.layers.flatten(prev_layer)
    flat = prev_layer

    prev_layer = noisy_dense(
        inputs = prev_layer,
        units = 256,
        activation = tf.nn.relu,
        name = 'dense_1',
        scope_name = "conv"
    )


    prev_layer = noisy_dense(
        inputs = prev_layer,
        units = 256,
        activation = tf.nn.relu,
        name = 'dense_2',
        scope_name = "conv"
    )

    logits = noisy_dense(
        inputs = prev_layer,
        units = 10,
        activation = lambda t:t,
        name = 'dense_3',
        scope_name = "conv"
    )


# In[23]:


class Generator():
    def __init__(self, imgsize, conv_input, logits, cls_no=10):
        imgshape = (imgsize, imgsize, 3)
        self.x = tf.placeholder(tf.float32, imgshape)
        self.y_adv = tf.placeholder(tf.int32, ())
        self.x_adv = conv_input

        self.assign_op = tf.assign(self.x_adv, self.x)

        self.lr = tf.placeholder(tf.float32, ())
        labels = tf.one_hot(self.y_adv, cls_no)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
        self.optim_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, var_list=[self.x_adv])

        self.epsilon = tf.placeholder(tf.float32, ())
        below = self.x - self.epsilon
        above = self.x + self.epsilon
        projected = tf.clip_by_value(tf.clip_by_value(self.x_adv, below, above), 0, 1)
        with tf.control_dependencies([projected]):
            self.project_step = tf.assign(self.x_adv, projected)

    def generate(self, sess, image, target, eps_val=0.01, lr_val=1e-1, num_steps=100):
        sess.run(self.assign_op, feed_dict={self.x: image})

        for i in range(num_steps):
            _, loss_val = sess.run([self.optim_step, self.loss], feed_dict={self.lr: lr_val, self.y_adv: target})
            sess.run(self.project_step, feed_dict={self.x: image, self.epsilon: eps_val})
            if (i + 1) % 10 == 0:
                print('step %d, loss=%g' % (i+1, loss_val))

        return sess.run(self.x_adv)


# In[9]:


pred = tf.nn.softmax(logits)
loss=ce(logits, y_)
accuracy=acc(pred, y_)

theta_conv = tf.trainable_variables("conv")
conv_solver = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss = loss)

fgsm_agent = Generator(32, x, logits)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_itr.initializer)


# In[10]:


target_label = 0


# In[21]:


saver = tf.train.Saver(var_list = theta_conv)
saver.restore(sess,"new_pretrained/model")

i = 0

adv_imgs = None
for each in zip(x_test,y_test):
    if sess.run(accuracy, feed_dict={x: each[0],y_: [each[1]],mode: EVAL}) == 1 and np.argmax(each[1])!=target_label:
        adv_img = np.expand_dims(fgsm_agent.generate(sess,each[0],target_label), axis=0)
        if adv_imgs is None:
            adv_imgs = adv_img
        else:
            adv_imgs = np.concatenate((adv_imgs,adv_img),axis = 0)
        i+= 1
    if i >= 100:
        break
print(adv_imgs.shape)


# In[20]:


import pickle
with open("adv_imgs.pkl",'wb') as file:
    pickle.dump(adv_imgs,file)


# In[22]:


FINAL_ACC=0.
for i in range(0,100):
    FINAL_ACC+=1/100*sess.run(accuracy, feed_dict={x: adv_imgs[i], y_: [np.eye(10)[target_label]], mode: EVAL})   
print("Final accuracy on test set:", FINAL_ACC)


# In[11]:


X = None
for i in range(0,1):
    if i == 0:
        X = sess.run(flat, feed_dict={x: x_test[i*1000:(i+1)*1000], y_: y_test[i*1000:(i+1)*1000], mode: EVAL}) 
    else:
        X = np.concatenate((X,sess.run(flat, feed_dict={x: x_test[i*1000:(i+1)*1000], y_: y_test[i*1000:(i+1)*1000], mode: EVAL})), axis = 0)
    print(X.shape)


# In[12]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
colors = [i*25 for i in np.argmax(y_test[0:1000], axis = 1)]

embedding = TSNE(n_components=2).fit_transform(X)
plt.scatter(embedding[:,0],embedding[:,1], c = colors )


# In[3]:


import numpy as np
for each in zip(np.array([1,2,3]),np.array([1,2,3])):
    print(each)


# In[10]:


targeted_label = np.array([i for i in range(10)])
np.random.shuffle(targeted_label)
print(targeted_label)

