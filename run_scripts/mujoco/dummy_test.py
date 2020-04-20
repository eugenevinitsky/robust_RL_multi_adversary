from copy import deepcopy
import errno
from datetime import datetime
import os
import subprocess
import sys

import pytz
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import Trainable
from ray.tune.logger import pretty_print
from ray.tune import run as run_tune
from ray.tune.registry import register_env


from visualize.mujoco.transfer_tests import run_transfer_tests
from visualize.mujoco.visualize_adversaries import visualize_adversaries
from utils.parsers import init_parser, ray_parser, ma_env_parser
from utils.pendulum_env_creator import lerrel_pendulum_env_creator
from utils.rllib_utils import get_config_from_path

from models.recurrent_tf_model_v2 import LSTM

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box
import random
import numpy as np

class DummyEnv(MultiAgentEnv):
    """Simple env in which the policy learns to repeat a previous observation
    token after a given delay."""

    def __init__(self):
        self.observation_space = Box(low=-1, high=1, shape=(2,))
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.step_num = 0

    def reset(self):
        self.step_num = 0.0
        return {'agent': np.zeros(2), 'adversary': np.zeros(2)}

    def step(self, action):
        self.step_num += 1
        if self.step_num > 100:
            done = {'__all__': True}
        else:
            done = {'__all__': False}
        return {'agent': np.zeros(2), 'adversary': np.zeros(2)}, {'agent': 100, 'adversary': 100}, done, {}


class AlternateTraining(Trainable):
    def _setup(self, config):
        self.config = config
        self.env = config['env']
        agent_config = self.config
        adv_config = deepcopy(self.config)
        agent_config['multiagent']['policies_to_train'] = ['agent']
        adv_config['multiagent']['policies_to_train'] = ['adversary']

        self.agent_trainer = PPOTrainer(env=self.env, config=agent_config)
        self.adv_trainer = PPOTrainer(env=self.env, config=adv_config)

    def _train(self):
        # improve the Adversary policy
        print("-- Adversary Training --")
        original_weight = self.adv_trainer.get_weights(["adversary"])['adversary']['adversary/fc_1/kernel'][0, 0]
        print(pretty_print(self.adv_trainer.train()))
        first_weight = self.adv_trainer.get_weights(["adversary"])['adversary']['adversary/fc_1/kernel'][0, 0]

        # Check that the weights are updating after training
        assert original_weight != first_weight, 'The weight hasn\'t changed after training what gives'

        # swap weights to synchronize
        self.agent_trainer.set_weights(self.adv_trainer.get_weights(["adversary"]))

        # improve the Agent policy
        print("-- Agent Training --")
        output = self.agent_trainer.train()

        # Assert that the weight hasn't changed but it has
        new_weight = self.agent_trainer.get_weights(["adversary"])['adversary']['adversary/fc_1/kernel'][0, 0]

        # Check that the adversary is not being trained when the agent trainer is training
        assert first_weight == new_weight, 'The weight of the adversary matrix has changed but it shouldnt have been updated!'

        # swap weights to synchronize
        self.adv_trainer.set_weights(self.agent_trainer.get_weights(["agent"]))

        return output

    def _save(self, tmp_checkpoint_dir):
        return self.agent_trainer._save(tmp_checkpoint_dir)


if __name__ == '__main__':
    env = DummyEnv()
    config = DEFAULT_CONFIG
    config['train_batch_size'] = 500
    config['lr'] = 1.0
    policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, {}),
                     'adversary': (PPOTFPolicy, env.observation_space, env.action_space, {})}

    print("========= Policy Graphs ==========")
    print(policy_graphs)

    policies_to_train = ['agent', 'adversary']


    def policy_mapping_fn(agent_id):
        return agent_id

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': policy_mapping_fn,
            'policies_to_train': policies_to_train
        }
    })

    config['env'] = 'DummyEnv'
    env_creator = lambda config: DummyEnv()
    register_env('DummyEnv', env_creator)

    exp_dict = {
        'name': 'ProofOfConcept',
        'run_or_experiment': AlternateTraining,
        'stop': {
            'training_iteration': 100
        },
        'config': config,
    }

    ray.init(local_mode=True)
    run_tune(**exp_dict, queue_trials=False)