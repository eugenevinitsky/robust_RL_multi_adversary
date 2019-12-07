from gym import Env
from gym.spaces import Box

import numpy as np

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

class DummyEnv(Env):

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(20, ))

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=(2,))

    def step(self, action):
        return np.zeros(self.observation_space.shape), 1, False, {}

    def reset(self):
        return np.zeros(self.observation_space.shape)

def env_creator(env_config):
    return DummyEnv()

if __name__=='__main__':
    ray.init(object_store_memory=(5e9), redis_max_memory=int(1e9), memory=int(5e9))

    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    register_env('DummyEnv', env_creator)

    config['env'] = 'CrowdSim'
    config['num_workers'] = 2
    config['lr'] = tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6])
    config['train_batch_size'] = 3000
    register_env('CrowdSim', env_creator)


    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)


    exp_dict = {
        'name': 'Test',
        'run_or_experiment': alg_run,
        'checkpoint_freq': 1000,
        'stop': {
            'training_iteration': 10000
        },
        'config': config,
    }

    run_tune(**exp_dict, queue_trials=False)

