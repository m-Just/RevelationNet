import tensorflow as tf

class Generator():
    def __init__(self, imgsize, conv_input, logits, targeted=True, cls_no=10):
        imgshape = (imgsize, imgsize, 3)
        self.x = tf.placeholder(tf.float32, imgshape)
        self.y_adv = tf.placeholder(tf.int32, ())
        self.x_adv = conv_input
        self.targeted = targeted
        self.epsilon = tf.placeholder(tf.float32, ())

        self.assign_op = tf.assign(self.x_adv, self.x)

        labels = tf.one_hot(self.y_adv, cls_no)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=[labels])
        if self.targeted:
            min_objective = self.loss
        else:
            min_objective = -self.loss

        # Using adaptive learning rate for FGM
        gradient, _ = tf.train.GradientDescentOptimizer(1.)\
            .compute_gradients(min_objective, var_list=[self.x_adv])[0]
        lr = self.epsilon / tf.norm(gradient, ord=2)
        #lr = self.epsilon / tf.reduce_mean(self.loss)
        self.optim_step = tf.train.GradientDescentOptimizer(lr).minimize(min_objective, var_list=[self.x_adv])
        
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        #sign = tf.sign(optimizer.compute_gradients(min_objective, var_list=[self.x_adv])[0][0]) * self.epsilon
        #self.optim_step = tf.assign(self.x_adv, sign + self.x_adv)

        below = self.x - self.epsilon
        above = self.x + self.epsilon
        projected = tf.clip_by_value(tf.clip_by_value(self.x_adv, below, above), 0, 1)
        with tf.control_dependencies([projected]):
            self.project_step = tf.assign(self.x_adv, projected)

    def generate(self, sess, image, target, eps_val=0.01, num_steps=100):
        adv_imgs = []

        modified = image
        for i in range(num_steps):
            sess.run(self.assign_op, feed_dict={self.x: modified})
            _, loss_val = sess.run([self.optim_step, self.loss], feed_dict={self.y_adv: target, self.epsilon: eps_val})
            sess.run(self.project_step, feed_dict={self.x: image, self.epsilon: eps_val})
            if (i + 1) % 10 == 0:
                print('step %d, loss=%g' % (i+1, loss_val))
            modified = sess.run(self.x_adv)
            adv_imgs.append(modified)

        return adv_imgs
