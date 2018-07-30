import tensorflow as tf

class Generator():
    def __init__(self, method, imgsize, conv_input, logits, targeted=True, cls_no=10, optimize_method='sgd'):
        imgshape = (imgsize, imgsize, 3)
        self.x = tf.placeholder(tf.float32, imgshape)
        self.y_adv = tf.placeholder(tf.int32, ())
        self.x_adv = conv_input
        self.targeted = targeted
        self.epsilon = tf.placeholder(tf.float32, ())
        self.step_scale = tf.placeholder(tf.float32, ())
        self.num_steps = tf.placeholder(tf.float32, ())

        self.assign_op = tf.assign(self.x_adv, self.x)

        labels = tf.one_hot(self.y_adv, cls_no)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=[labels])
        if self.targeted:
            min_objective = self.loss
        else:
            min_objective = -self.loss

        print('Using %s attack' % method)
        if optimize_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(1.)
        elif optimize_method == 'momentum':
            optimizer = tf.train.MomentumOptimizer(1., 0.9, use_nesterov=True)
        elif optimize_method == 'adam':
            optimizer = tf.train.AdamOptimizer(1.)
        self.optimizer = optimizer

        if method == 'FG':
            gradient, var = optimizer.compute_gradients(
                min_objective, var_list=[self.x_adv])[0]
            step_size = self.step_scale * self.epsilon / tf.norm(gradient, ord=2)
            grads_and_vars = [(step_size * gradient, var)]
            self.optim_step = optimizer.apply_gradients(grads_and_vars)
            #self.optim_step = tf.train.GradientDescentOptimizer(step_size)\
            #    .minimize(min_objective, var_list=[self.x_adv])

        elif method == 'FGS':
            gradient, _ = optimizer.compute_gradients(
                -min_objective, var_list=[self.x_adv])[0]
            l2_norm = (imgsize * imgsize * 3) ** 0.5
            step_size = self.step_scale * self.epsilon / l2_norm
            pert = tf.sign(gradient) * step_size
            self.optim_step = tf.assign_add(self.x_adv, pert)

        below = self.x - self.epsilon
        above = self.x + self.epsilon
        projected = tf.clip_by_value(tf.clip_by_value(self.x_adv, below, above), 0, 1)
        with tf.control_dependencies([projected]):
            self.project_step = tf.assign(self.x_adv, projected)

    def generate(self, sess, image, target, eps_val=0.01, step_scale=1., num_steps=100, loss_thresh=None):
        sess.run(tf.variables_initializer(self.optimizer.variables()))

        adv_imgs = []
        modified = image
        for i in range(num_steps):
            sess.run(self.assign_op, feed_dict={self.x: modified})
            _, loss_val = sess.run([self.optim_step, self.loss],
                feed_dict={self.y_adv: target,
                           self.epsilon: eps_val,
                           self.step_scale: step_scale,
                           self.num_steps: num_steps})
                
            sess.run(self.project_step, feed_dict={self.x: image, self.epsilon: eps_val})
            if (i + 1) % 10 == 0:
                print('  step %d, loss=%g' % (i+1, loss_val))
            modified = sess.run(self.x_adv)
            adv_imgs.append(modified)

            if loss_thresh is not None:
                if loss_val > loss_thresh: break

        return adv_imgs
