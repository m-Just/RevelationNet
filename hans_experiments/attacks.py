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
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'delta': self.np_dtype}
        self.structural_kwargs = ['ord']

    def generate(self, x, **kwargs):
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        grad_batch = tf.Variable(tf.zeros(x.get_shape()), trainable=False)
        _, h, w, c = x.get_shape().as_list()
        for h, w, c in product(range(h), range(w), range(c)):
            basis = np.zeros([h, w, c], dtype=np.float32)
            basis[h, w, c] = self.delta

            pos = x + tf.constant(basis)
            pos_preds = self.model.get_probs(pos)

            neg = x - tf.constant(basis)
            neg_preds = self.model.get_probs(neg)
            
            grads = (pos_preds - neg_preds) / (2 * self.delta)
            grad_batch[:, h, w, c] = grads

            if self.y_target is not None: grad_batch = -grad_batch

        adv_x = x + self.eps * grad_batch

        if (self.clip_min is not None) and (self.clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x
        
    def parse_params(self, eps=0.3, delta=1e-6, ord=np.inf, y=None, y_target=None,
                     clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) attack step size (input variation)
        :param delta: (optional float) it controls the accuracy of estimation
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters

        self.eps = eps
        self.delta = delta
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True
