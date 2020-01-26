"""A distribution that returns the mean and std deviation of a diagonal gaussian"""

import numpy as np
from ray.rllib.models.tf.tf_action_dist import DiagGaussian, TFActionDistribution, ActionDistribution
from ray.rllib.utils.annotations import override
import tensorflow as tf

class LogitsDist(DiagGaussian):

    @override(TFActionDistribution)
    def _build_sample_op(self):
        return tf.concat([self.mean, self.std,
                          self.mean + self.std * tf.random_normal(tf.shape(self.mean))], axis=-1)

    @override(ActionDistribution)
    def logp(self, x):
        import ipdb; ipdb.set_trace()
        sample = x[-2:]
        return (-0.5 * tf.reduce_sum(
            tf.square((sample - self.mean) / self.std), reduction_indices=[1]) -
                0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) -
                tf.reduce_sum(self.log_std, reduction_indices=[1]))

    @override(ActionDistribution)
    def kl(self, other):
        assert isinstance(other, DiagGaussian)
        return tf.reduce_sum(
            other.log_std - self.log_std +
            (tf.square(self.std) + tf.square(self.mean - other.mean)) /
            (2.0 * tf.square(other.std)) - 0.5,
            reduction_indices=[1])