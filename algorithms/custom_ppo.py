import logging
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.agents.ppo.ppo_policy import ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo_policy import vf_preds_and_logits_fetches
from ray.rllib.agents.ppo.ppo_policy import kl_and_loss_stats

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG as DEFAULT_PPO_CONFIG
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, setup_mixins, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.tf_policy import ACTION_LOGP


# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"

DEFAULT_CONFIG = DEFAULT_PPO_CONFIG
# DEFAULT_CONFIG.update({
#     "num_adversaries": 2,
#     # Initial weight on the kl diff part of the loss
#     "kl_diff_weight": 1.0,
#     # Target KL between agents
#     "kl_diff_target": 1.0
# })


def get_logits(model, train_batch, index):
    """
    This method is used to compute the logits that the model would have on the adversary batch of agent index.
    That is, if model had been deployed instead of adversary `index`, this computes the logits we would have gotten
    in that case.
    :param model: an RLlib model object
    :param train_batch: (dict)
        The observations of every adversary is available in the observations via <Dict Key>_<Adversary Index>
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

    batch_size = sample_batch['obs'].shape[0]

    # We store the observations of all of the other agents in postprocess as well as their corresponding logits.
    # Then, we can take our agent and run its model on all of these observations. This gives us the logits of our
    # agent had it seen these observations. We can then compute the KL between the two policies.
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

    # handle the fake pass. There aren't any other_agent_batches in the rllib fake pass
    if not other_agent_batches:
        # go through and fill in as many kl objects as you will need
        for i in range(policy.num_adversaries - 1):
            action_dim = sample_batch[SampleBatch.PREV_ACTIONS].shape[1]
            postprocess["kj_log_std_{}".format(i)] = np.zeros((batch_size, action_dim)).astype(np.float32)
            postprocess["kj_std_{}".format(i)] = np.zeros((batch_size, action_dim)).astype(np.float32)
            postprocess["kj_mean_{}".format(i)] = np.zeros((batch_size, action_dim)).astype(np.float32)
            postprocess[SampleBatch.CUR_OBS + '_' + str(i)] = sample_batch[SampleBatch.CUR_OBS]
            postprocess[SampleBatch.PREV_ACTIONS + '_' + str(i)] = sample_batch[SampleBatch.PREV_ACTIONS]
            postprocess[SampleBatch.PREV_REWARDS + '_' + str(i)] = sample_batch[SampleBatch.PREV_REWARDS]


    return postprocess


def setup_kl_loss(policy, model, dist_class, train_batch):
    """Since we are computing the logits of model on the observations of adversary i, we compute
       the KL as \mathbb{E}_{adversary i} [log(p(adversary_i) \ p(model)] instead of the other way around."""
    kl_loss = 0.0
    for i in range(policy.num_adversaries - 1):
        logits, state = get_logits(model, train_batch, i)
        mean, log_std = tf.split(logits, 2, axis=1)
        std = tf.exp(log_std)
        other_logstd = train_batch["kj_log_std_{}".format(i)]
        other_std = train_batch["kj_std_{}".format(i)]
        other_mean = train_batch["kj_mean_{}".format(i)]

        # we clip here lest it blow up due to some really small probabilities
        kl_loss += tf.losses.huber_loss(tf.reduce_sum(
            - other_logstd + log_std +
            (tf.square(other_std) + tf.square(mean - other_mean)) /
            (2.0 * tf.square(std)) - 0.5,
            axis=1
        ), policy.kl_target)
    return kl_loss


# def new_ppo_surrogate_loss(policy, batch_tensors):
def new_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    if policy.num_adversaries > 1:
        kl_diff_loss = setup_kl_loss(policy, model, dist_class, train_batch)

    # zero out the loss elements where you weren't actually acting
    original_space = restore_original_dimensions(train_batch['obs'], model.obs_space)
    is_active = original_space['is_active']

    # extract the ppo_surrogate_loss before the mean is taken
    ppo_custom_surrogate_loss(policy, model, dist_class, train_batch)
    pre_mean_loss = policy.loss_obj.pre_mean_loss

    def reduce_mean_valid(t):
        return tf.reduce_mean(tf.boolean_mask(t, policy.loss_obj.valid_mask))

    # This mask combines both the valid mask and a check for when we were actually active in the env
    combined_mask = tf.math.logical_and(policy.loss_obj.valid_mask, tf.cast(tf.squeeze(is_active, -1), tf.bool))
    standard_loss = tf.reduce_mean(tf.boolean_mask(pre_mean_loss, combined_mask))

    # Since we are happy to evaluate the kl diff over obs in which we weren't active, we only mask this
    # with respect to the valid mask, which tracks padding for RNNs
    if policy.num_adversaries > 1 and policy.config['kl_diff_weight'] > 0:
        policy.unscaled_kl_loss = kl_diff_loss
        clipped_mean_loss = reduce_mean_valid(tf.clip_by_value(kl_diff_loss, 0, policy.kl_diff_clip))
        policy.kl_var = tf.math.reduce_std(kl_diff_loss)
        return -policy.config['kl_diff_weight'] * clipped_mean_loss + standard_loss
    else:
        return standard_loss
    # return reduce_mean_valid(pre_mean_loss * tf.squeeze(is_active))


def new_kl_and_loss_stats(policy, train_batch):
    """Add the kl stats to the fetches"""
    stats = kl_and_loss_stats(policy, train_batch)
    if policy.num_adversaries > 1:
        info = {'kl_diff': policy.unscaled_kl_loss,
                "cur_kl_diff_coeff": tf.cast(policy.kl_diff_coeff, tf.float64),
                'kl_diff_var': policy.kl_var
                }
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


class KLDiffMixin(object):
    def __init__(self, config):
        # KL Coefficient
        self.kl_diff_coeff_val = config["kl_diff_weight"]
        self.kl_target = config["kl_diff_target"]
        self.kl_diff_clip = config["kl_diff_clip"]
        # self.kl_diff_coeff = tf.compat.v1.get_variable(
        #     initializer=tf.constant_initializer(self.kl_diff_coeff_val),
        #     name="kl_coeff_diff",
        #     shape=(),
        #     trainable=False,
        #     dtype=tf.float32)
        self.kl_diff_coeff = tf.Variable(self.kl_diff_coeff_val,
            name="kl_coeff_diff",
            trainable=False,
            dtype=tf.float32)

    def update_kl_diff(self, sampled_kl):
        if sampled_kl < 2.0 * self.kl_target:
            self.kl_diff_coeff_val *= 1.5
        elif sampled_kl > 0.5 * self.kl_target:
            self.kl_diff_coeff_val *= 0.5
        # There is literally no reason to let this go off to infinity and subsequently overflow
        self.kl_diff_coeff_val = max(self.kl_diff_coeff_val, 1e5)
        self.kl_diff_coeff.load(self.kl_diff_coeff_val, session=self.get_session())
        return self.kl_diff_coeff_val


class SetUpConfig(object):
    def __init__(self, config):
        self.num_adversaries = config['num_adversaries']


def special_setup_mixins(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    SetUpConfig.__init__(policy, config)
    KLDiffMixin.__init__(policy, config)


def update_kl(trainer, fetches):
    """Update both the KL coefficient and the kl diff coefficient"""
    if "kl_diff" in fetches:
        # single-agent
        trainer.workers.local_worker().for_policy(
            lambda pi: pi.update_kl_difft(fetches["kl_diff"]))
    else:
        def update(pi, pi_id):
            # The robot won't have kl_diff in fetches
            if pi_id in fetches and "kl_diff" in fetches[pi_id]:
                pi.update_kl_diff(fetches[pi_id]["kl_diff"])
            else:
                logger.debug("No data for {}, not updating kl_diff".format(pi_id))

        # multi-agent
        trainer.workers.local_worker().foreach_trainable_policy(update)

    # We disable the KL update since this fights against the adversary diff. Might be worth not doing.

    # if "kl" in fetches:
    #     # single-agent
    #     trainer.workers.local_worker().for_policy(
    #         lambda pi: pi.update_kl(fetches["kl"]))
    # else:
    #
    #     def update(pi, pi_id):
    #         if pi_id in fetches:
    #             pi.update_kl(fetches[pi_id]["kl"])
    #         else:
    #             logger.debug("No data for {}, not updating kl".format(pi_id))
    #
    #     # multi-agent
    #     trainer.workers.local_worker().foreach_trainable_policy(update)


CustomPPOPolicy = PPOTFPolicy.with_updates(
    name="POPO",
    loss_fn=new_ppo_surrogate_loss,
    postprocess_fn=new_postprocess_ppo_gae,
    stats_fn=new_kl_and_loss_stats,
    extra_action_fetches_fn=new_vf_preds_and_logits_fetches,
    mixins=[SetUpConfig, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            ValueNetworkMixin, KLDiffMixin],
    before_loss_init=special_setup_mixins
)

KLPPOTrainer = PPOTrainer.with_updates(
    default_policy=CustomPPOPolicy,
    default_config=DEFAULT_CONFIG,
    after_optimizer_step=update_kl
)
