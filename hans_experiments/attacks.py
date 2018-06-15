from __future__ import division

from itertools import product

import numpy as np

from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack

class FiniteDifferenceMethod(Attack):
    
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(FiniteDifferenceMethod, self).__init__(model, back, sess, dtypestr)

    def generate_np(self, x_val, y_val, y_target=None, ord=np.inf,
                    eps=0.3, delta=1e-3, clip_min=0. clip_max=1.):
        assert isinstance(x_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        if y_target is not None:
            assert isinstance(y_target, np.ndarray)

        # construct graph
        x_shape = x_val.shape
        B, H, W, C = x_shape
        if B <= 128:
            raise ValueError('Large batch size may cause OOM, no more 
                             than 128 is recommended')

        x = tf.placeholder(tf.float32, x_shape)
        adv_target = y_val if y_target is None else y_target
        prob = tf.reduce_sum(x * tf.constant(adv_target), axis=1)

        # generate adversarial samples
        grad_est = np.zeros(x_shape, dtype=np.float32)
        basis = np.zeros([H, W, C], dtype=np.float32)
        for h, w, c in product(range(H), range(W), range(C)):
            basis[h, w, c] = delta

            pos_prob = self.sess.run(prob, feed_dict={x: x_val + basis})
            neg_prob = self.sess.run(prob, feed_dict={x: x_val - basis})

            if ord == np.inf:
                grad = eps * np.sign(pos_prob - neg_prob)
            else:
                raise NotImplementedError('Only L-inf is currently implemented')

            if y_target is not None: grad = -grad
            grad_est[:, h, w, c] = grad

            basis[h, w, c] = 0.

        adv_x = x_val + grad_est
        if (clip_min is not None) and (clip_max is not None):
            adv_x = np.clip_by_value(adv_x, clip_min, clip_max)

        return adv_x
