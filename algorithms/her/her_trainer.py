from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from algorithms.her.her_optimizer import HEROptimizer
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer, \
    update_worker_explorations
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, \
    DEFAULT_CONFIG as DDPG_CONFIG
from ray.rllib.utils import merge_dicts
# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = merge_dicts(
    DDPG_CONFIG,
    {
        # largest changes: twin Q functions, delayed policy updates, and target
        # smoothing
        "twin_q": True,
        "policy_delay": 2,
        "smooth_target_policy": True,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,

        # other changes & things we want to keep fixed: IID Gaussian
        # exploration noise, larger actor learning rate, no l2 regularisation,
        # no Huber loss, etc.
        "exploration_should_anneal": False,
        "exploration_noise_type": "gaussian",
        "exploration_gaussian_sigma": 0.1,
        "learning_starts": 10000,
        "pure_exploration_steps": 10000,
        "actor_hiddens": [400, 300],
        "critic_hiddens": [400, 300],
        "n_step": 1,
        "gamma": 0.99,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "l2_reg": 0.0,
        "tau": 5e-3,
        "train_batch_size": 100,
        "use_huber": False,
        "target_network_update_freq": 0,
        "num_workers": 0,
        "num_gpus_per_worker": 0,
        "per_worker_exploration": False,
        "worker_side_prioritization": False,
        "buffer_size": 1000000,
        "prioritized_replay": False,
        "clip_rewards": False,
        "use_state_preprocessor": False,
    },
)
# __sphinx_doc_end__
# yapf: enable


def make_exploration_schedule(config, worker_index):
    # Modification of DQN's schedule to take into account
    # `exploration_ou_noise_scale`
    if config["per_worker_exploration"]:
        assert config["num_workers"] > 1, "This requires multiple workers"
        if worker_index >= 0:
            # FIXME: what do magic constants mean? (0.4, 7)
            max_index = float(config["num_workers"] - 1)
            exponent = 1 + worker_index / max_index * 7
            return ConstantSchedule(0.4**exponent)
        else:
            # local ev should have zero exploration so that eval rollouts
            # run properly
            return ConstantSchedule(0.0)
    elif config["exploration_should_anneal"]:
        return LinearSchedule(
            schedule_timesteps=int(config["exploration_fraction"] *
                                   config["schedule_max_timesteps"]),
            initial_p=1.0,
            final_p=config["exploration_final_scale"])
    else:
        # *always* add exploration noise
        return ConstantSchedule(1.0)


def setup_ddpg_exploration(trainer):
    trainer.exploration0 = make_exploration_schedule(trainer.config, -1)
    trainer.explorations = [
        make_exploration_schedule(trainer.config, i)
        for i in range(trainer.config["num_workers"])
    ]


def add_pure_exploration_phase(trainer):
    global_timestep = trainer.optimizer.num_steps_sampled
    pure_expl_steps = trainer.config["pure_exploration_steps"]
    if pure_expl_steps:
        # tell workers whether they should do pure exploration
        only_explore = global_timestep < pure_expl_steps
        trainer.workers.local_worker().foreach_trainable_policy(
            lambda p, _: p.set_pure_exploration_phase(only_explore))
        for e in trainer.workers.remote_workers():
            e.foreach_trainable_policy.remote(
                lambda p, _: p.set_pure_exploration_phase(only_explore))
    update_worker_explorations(trainer)


def make_optimizer(workers, config):
    return HEROptimizer(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        prioritized_replay=config["prioritized_replay"],
        prioritized_replay_alpha=config["prioritized_replay_alpha"],
        prioritized_replay_beta=config["prioritized_replay_beta"],
        schedule_max_timesteps=config["schedule_max_timesteps"],
        beta_annealing_fraction=config["beta_annealing_fraction"],
        final_prioritized_replay_beta=config["final_prioritized_replay_beta"],
        prioritized_replay_eps=config["prioritized_replay_eps"],
        train_batch_size=config["train_batch_size"],
        sample_batch_size=config["sample_batch_size"],
        **config["optimizer"])


HERTrainer = GenericOffPolicyTrainer.with_updates(
    name="HER",
    default_config=DEFAULT_CONFIG,
    default_policy=DDPGTFPolicy,
    before_init=setup_ddpg_exploration,
    before_train_step=add_pure_exploration_phase,
    make_policy_optimizer=make_optimizer,
)
