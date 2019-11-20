from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ppo.ppo_policy import ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo_policy import vf_preds_and_logits_fetches
from ray.rllib.agents.ppo.ppo_policy import kl_and_loss_stats

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, setup_mixins, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin

import numpy as np
import tensorflow as tf


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
        for i, other_agent_batch in enumerate(other_agent_batches.values()):
            try:
                batch = other_agent_batch[1]
                postprocess["kj_log_std_{}".format(i)] = batch["kj_log_std"]
                postprocess["kj_std_{}".format(i)] = batch["kj_std"]
                postprocess["kj_mean_{}".format(i)] = batch["kj_mean"]#np.zeros((batch_size, 1))
                postprocess[SampleBatch.CUR_OBS + '_' + str(i)] = sample_batch[SampleBatch.CUR_OBS]
                postprocess[SampleBatch.PREV_ACTIONS + '_' + str(i)] = sample_batch[SampleBatch.PREV_ACTIONS]
                postprocess[SampleBatch.PREV_REWARDS + '_' + str(i)] = sample_batch[SampleBatch.PREV_REWARDS]

            except:
                import ipdb; ipdb.set_trace()


    # handle the fake pass
    if not other_agent_batches:
        # go through and fill in as many kl objects as you will need
        # import ipdb; ipdb.set_trace()
        for i in range(policy.num_adversaries):
            postprocess["kj_log_std_{}".format(i)] = np.zeros((batch_size, 2))
            postprocess["kj_std_{}".format(i)] = np.zeros((batch_size, 2))
            postprocess["kj_mean_{}".format(i)] = np.zeros((batch_size, 2))
            postprocess[SampleBatch.CUR_OBS + '_' + str(i)] = sample_batch[SampleBatch.CUR_OBS]
            postprocess[SampleBatch.PREV_ACTIONS + '_' + str(i)] = sample_batch[SampleBatch.PREV_ACTIONS]
            postprocess[SampleBatch.PREV_REWARDS + '_' + str(i)] = sample_batch[SampleBatch.PREV_REWARDS]

    return postprocess
        

def setup_kl_loss(policy, model, dist_class, train_batch):


    kl_loss = 0.0
    for i in range(policy.num_adversaries):
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
    return ppo_surrogate_loss(policy, model, dist_class, train_batch) \
    + policy.kl_diff_loss


def new_kl_and_loss_stats(policy, train_batch):
    # import ipdb; ipdb.set_trace()
    # total_kl_diff = sum([batch['kl_diff'] for batch in batch_tensors])
    info = {'kl_diff': policy.kl_diff_loss}
    stats = kl_and_loss_stats(policy, train_batch)
    stats.update(info)
    return stats


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
    