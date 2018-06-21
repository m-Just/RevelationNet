import os
import numpy as np
import tensorflow as tf
from utilities import Target
from keras.preprocessing.image import ImageDataGenerator



TRAIN = 0
EVAL = 1

class Model():
    def __init__(self, noise_infos, init_path = None):
        assert len(noise_infos) == 7
        self.init_path = init_path
        self.noise_infos = noise_infos
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32,[None,32,32,3])
        self.y = tf.placeholder(tf.float32,[None,10])
        self.mode = tf.placeholder(tf.int32,[])
        self.logits, self.pred, self.loss, self.acc = Target(self.x,self.y,self.mode,noise_infos)
        self.theta = tf.trainable_variables("conv")
        self.saver = tf.train.Saver(var_list = self.theta)
        
        self.sess.run(tf.global_variables_initializer())
        if init_path is not None:
            self.saver.restore(self.sess,init_path)
        
    def reset(self,noise_infos = None, init_path = None):
        tf.reset_default_graph()
        if noise_infos == None:
            noise_infos = self.noise_infos
        if init_path == None:
            init_path = self.init_path
        self.__init__(noise_infos, init_path)
        
    def train(self,x_train,y_train, x_test, y_test, batch_size = 128, epoch = 50, learning_rate = 0.001, weight_decay = None, optimizer ="adam", init_path = None, save_path = None,  augmentation = True):
        
        #learning_rate = tf.placeholder(tf.float32, shape=[])
        if weight_decay is not None:
            reg_var = [each for each in self.theta if "kernel" in each.name]
            reg_loss = [tf.nn.l2_loss(var) for var in reg_var]
            reg_loss = weight_decay * tf.add_n(reg_loss)
            loss = self.loss + reg_loss
        else:
            loss = self.loss
        print(self.theta)
        if optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss,var_list = self.theta)
        elif optimizer == "rms":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay=1e-6, momentum=0.9).minimize(loss,var_list = self.theta)
        elif optimizer == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss,var_list = self.theta)
        elif optimizer == "moment":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,momentum = 0.9, use_nesterov = True).minimize(loss,var_list = self.theta)
        
        
        self.sess.run(tf.global_variables_initializer())
        if init_path is not None:
            self.saver.restore(self.sess,init_path)
          
        if augmentation:
            # This will do preprocessing and realtime data augmentation:
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
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            
        datagen.fit(x_train)
        flow = datagen.flow(x_train, y_train, batch_size=batch_size)
        
        i = 0
        itr = 50000 * epoch // batch_size 
        for each in flow:
            if i >= itr:
                break
            if i % 391 == 0:
                print(str(i//391) + " epoch: ", self.sess.run([self.loss,self.acc],feed_dict = {self.x:x_test[0:1000],self.y:y_test[0:1000],self.mode:EVAL}))
            i += 1
            self.sess.run(self.optimizer,feed_dict = {self.x:each[0],self.y:each[1],self.mode:TRAIN})
            
            
        if save_path is not None:
            self.saver.save(self.sess,save_path)
            
    def test(self,x_test,y_test, batch_size = 1000, init_path = None):
        if init_path is not None:
            sess.saver.restore(self.sess,init_path)
            
        itr = x_test.shape[0]// batch_size
        final_acc = 0
        for i in range(itr):
            final_acc += 1/itr * self.sess.run(self.acc,feed_dict = {self.x:x_test[i*batch_size:(i+1)*batch_size],self.y:y_test[i*batch_size:(i+1)*batch_size],self.mode:EVAL})
        return final_acc
            
        
            
        
        