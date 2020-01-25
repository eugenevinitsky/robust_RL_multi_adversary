"""A distribution that returns the mean and std deviation of a diagonal gaussian"""

from ray.rllib.models.tf.tf_action_dist import DiagGaussian, TFActionDistribution
from ray.rllib.utils.annotations import override
import tensorflow as tf

class LogitsDist(DiagGaussian):
    @override(TFActionDistribution)
    def _build_sample_op(self):
        return tf.concat([self.mean, self.std], axis=-1)