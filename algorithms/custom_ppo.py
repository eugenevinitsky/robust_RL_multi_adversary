from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.agents.ppo.ppo_policy import ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo_policy import vf_preds_and_logits_fetches
from ray.rllib.agents.ppo.ppo_policy import kl_and_loss_stats

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, setup_mixins, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.tf_policy import ACTION_LOGP

import numpy as np
import tensorflow as tf

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"


def get_logits(model, train_batch, index):
    """
    :param model: an RLlib model object
    :param train_batch: (dict)
    :param index: (int)
        The index of the agent whose batch we are iterating over
    :return: logits, state
        The logits and the hidden state corresponding to those logits
    """

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS + '_' + str(index)],
        "is_training": False,
    }
    if SampleBatch.PREV_ACTIONS in train_batch:
        input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS + '_' + str(index)]
    if SampleBatch.PREV_REWARDS in train_batch:
        input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS + '_' + str(index)]
    states = []
    i = 0
    # TODO(@evinitsky) currently the adversaries should NOT be LSTMs, I don't know how we handle that yet
    while "state_in_{}".format(i) in train_batch:
        states.append(train_batch["state_in_{}".format(i)])
        i += 1
    return model.__call__(input_dict, states, train_batch.get("seq_lens"))


def new_vf_preds_and_logits_fetches(policy):
    # new_info = {"action_dist": policy.action_dist}
    logits = policy.model.last_output()
    mean, log_std = tf.split(logits, 2, axis=1)
    new_info = {"kj_log_std": log_std,
                "kj_std": tf.exp(log_std),
                "kj_mean": mean}
    curr_dict = vf_preds_and_logits_fetches(policy)
    curr_dict.update(new_info)
    return curr_dict


def new_postprocess_ppo_gae(policy,
                            sample_batch,
                            other_agent_batches=None,
                            episode=None):
    postprocess = postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)

    # if other_agent_batches:
    batch_size = sample_batch['obs'].shape[0]
    if other_agent_batches:
        i = 0
        for other_agent_batch in enumerate(other_agent_batches.values()):
            batch = other_agent_batch[1][1]
            if "kj_log_std" in batch:
                postprocess["kj_log_std_{}".format(i)] = batch["kj_log_std"]
                postprocess["kj_std_{}".format(i)] = batch["kj_std"]
                postprocess["kj_mean_{}".format(i)] = batch["kj_mean"]  # np.zeros((batch_size, 1))
                postprocess[SampleBatch.CUR_OBS + '_' + str(i)] = sample_batch[SampleBatch.CUR_OBS]
                postprocess[SampleBatch.PREV_ACTIONS + '_' + str(i)] = sample_batch[SampleBatch.PREV_ACTIONS]
                postprocess[SampleBatch.PREV_REWARDS + '_' + str(i)] = sample_batch[SampleBatch.PREV_REWARDS]
                i += 1

    # handle the fake pass
    if not other_agent_batches:
        # go through and fill in as many kl objects as you will need
        for i in range(policy.num_adversaries - 1):
            action_dim = sample_batch[SampleBatch.PREV_ACTIONS].shape[1]
            postprocess["kj_log_std_{}".format(i)] = np.zeros((batch_size, action_dim))
            postprocess["kj_std_{}".format(i)] = np.zeros((batch_size, action_dim))
            postprocess["kj_mean_{}".format(i)] = np.zeros((batch_size, action_dim))
            postprocess[SampleBatch.CUR_OBS + '_' + str(i)] = sample_batch[SampleBatch.CUR_OBS]
            postprocess[SampleBatch.PREV_ACTIONS + '_' + str(i)] = sample_batch[SampleBatch.PREV_ACTIONS]
            postprocess[SampleBatch.PREV_REWARDS + '_' + str(i)] = sample_batch[SampleBatch.PREV_REWARDS]


    return postprocess


def setup_kl_loss(policy, model, dist_class, train_batch):
    kl_loss = 0.0
    for i in range(policy.num_adversaries - 1):
        logits, state = get_logits(model, train_batch, i)
        mean, log_std = tf.split(logits, 2, axis=1)
        std = tf.exp(log_std)
        other_logstd = train_batch["kj_log_std_{}".format(i)]
        other_std = train_batch["kj_std_{}".format(i)]
        other_mean = train_batch["kj_mean_{}".format(i)]

        kl_loss += tf.reduce_sum(
            other_logstd - log_std +
            (tf.square(std) + tf.square(mean - other_mean)) /
            (2.0 * tf.square(other_std)) - 0.5,
            reduction_indices=[1]
        )
    return -kl_loss


# def new_ppo_surrogate_loss(policy, batch_tensors):
def new_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    policy.kl_diff_loss = setup_kl_loss(policy, model, dist_class, train_batch)

    # zero out the loss elements where you weren't actually acting
    original_space = restore_original_dimensions(train_batch['obs'], model.obs_space)
    is_active = original_space['is_active']

    # TODO(@evinitsky) extract the ppo_surrogate_loss before the mean is taken
    ppo_custom_surrogate_loss(policy, model, dist_class, train_batch)
    pre_mean_loss = policy.loss_obj.pre_mean_loss

    def reduce_mean_valid(t):
        return tf.reduce_mean(tf.boolean_mask(t, policy.loss_obj.valid_mask))
    return reduce_mean_valid(pre_mean_loss * tf.squeeze(is_active) + policy.kl_diff_loss)


def new_kl_and_loss_stats(policy, train_batch):
    # import ipdb; ipdb.set_trace()
    # total_kl_diff = sum([batch['kl_diff'] for batch in batch_tensors])
    info = {'kl_diff': policy.kl_diff_loss}
    stats = kl_and_loss_stats(policy, train_batch)
    stats.update(info)
    return stats


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


class SetUpConfig(object):
    def __init__(self, config):
        self.num_adversaries = config['num_adversaries']


def special_setup_mixins(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    SetUpConfig.__init__(policy, config)


CustomPPOPolicy = PPOTFPolicy.with_updates(
    name="POPO",
    loss_fn=new_ppo_surrogate_loss,
    postprocess_fn=new_postprocess_ppo_gae,
    stats_fn=new_kl_and_loss_stats,
    extra_action_fetches_fn=new_vf_preds_and_logits_fetches,
    mixins=[SetUpConfig, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            ValueNetworkMixin],
    before_loss_init=special_setup_mixins
)

my_special_config = DEFAULT_CONFIG
my_special_config["num_adversaries"] = 2
KLPPOTrainer = PPOTrainer.with_updates(
    default_policy=CustomPPOPolicy,
    default_config=my_special_config
)
