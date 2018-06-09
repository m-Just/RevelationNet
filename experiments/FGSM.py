import tensorflow as tf

class Generator():
    def __init__(self, imgsize, conv_input, logits, cls_no=10):
        imgshape = (imgsize, imgsize, 3)
        self.x = tf.placeholder(tf.float32, imgshape)
        self.y_adv = tf.placeholder(tf.int32, ())
        self.x_adv = conv_input

        self.assign_op = tf.assign(self.x_adv, self.x)

        self.lr = tf.placeholder(tf.float32, ())
        labels = tf.one_hot(self.y_adv, cls_no)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=[labels])
        self.optim_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, var_list=[self.x_adv])

        self.epsilon = tf.placeholder(tf.float32, ())
        below = self.x - self.epsilon
        above = self.x + self.epsilon
        projected = tf.clip_by_value(tf.clip_by_value(self.x_adv, below, above), 0, 1)
        with tf.control_dependencies([projected]):
            self.project_step = tf.assign(self.x_adv, projected)

    def generate(self, sess, image, target, eps_val=0.01, lr_val=1e-1, num_steps=100):
        adv_imgs = []

        modified = image
        for i in range(num_steps):
            sess.run(self.assign_op, feed_dict={self.x: modified})
            _, loss_val = sess.run([self.optim_step, self.loss], feed_dict={self.lr: lr_val, self.y_adv: target})
            sess.run(self.project_step, feed_dict={self.x: image, self.epsilon: eps_val})
            if (i + 1) % 10 == 0:
                print('step %d, loss=%g' % (i+1, loss_val))
            modified = sess.run(self.x_adv)
            adv_imgs.append(modified)

        return adv_imgs
