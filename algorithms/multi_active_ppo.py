"""The purpose of this is to allow you to get the actions of multiple agents, but only pass gradients
through the one agent that is active at the time"""

from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

import ray
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.agents.ppo.ppo_policy import ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo_policy import vf_preds_and_logits_fetches
from ray.rllib.agents.ppo.ppo_policy import kl_and_loss_stats

from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG as DEFAULT_PPO_CONFIG
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, postprocess_ppo_gae, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin, clip_gradients, setup_config
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.tf_policy import ACTION_LOGP

from ray.rllib.agents.ppo.ppo import choose_policy_optimizer, validate_config, warn_about_bad_reward_scales, update_kl


# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"


def new_ppo_surrogate_loss(policy, model, dist_class, train_batch):

    # zero out the loss elements where you weren't actually acting
    original_space = restore_original_dimensions(train_batch['obs'], model.obs_space)
    is_active = original_space['is_active']

    # extract the ppo_surrogate_loss before the mean is taken
    ppo_custom_surrogate_loss(policy, model, dist_class, train_batch)
    pre_mean_loss = policy.loss_obj.pre_mean_loss

    # This mask combines both the valid mask and a check for when we were actually active in the env
    combined_mask = tf.math.logical_and(policy.loss_obj.valid_mask, tf.cast(tf.squeeze(is_active, -1), tf.bool))
    standard_loss = tf.reduce_mean(tf.boolean_mask(pre_mean_loss, combined_mask))

    return standard_loss


class PPOCustomLoss(object):
    def __init__(self,
                 action_space,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True,
                 model_config=None):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
            model_config (dict): (Optional) model config for use in specifying
                action distributions.
        """

        self.valid_mask = valid_mask
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            self.pre_mean_loss = -surrogate_loss + cur_kl_coeff * action_kl + \
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            self.pre_mean_loss = -surrogate_loss + cur_kl_coeff * action_kl - entropy_coeff * curr_entropy
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


def ppo_custom_surrogate_loss(policy, model, dist_class, train_batch):

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    policy.loss_obj = PPOCustomLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    return policy.loss_obj.loss



CustomPPOPolicy = build_tf_policy(
    name="PPOTFPolicy",
    get_default_config=lambda: deepcopy(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG),
    loss_fn=new_ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])


CustomPPOTrainer=build_trainer(
    name="MultiPPO",
    default_config=deepcopy(DEFAULT_PPO_CONFIG),
    default_policy=CustomPPOPolicy,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales
)
