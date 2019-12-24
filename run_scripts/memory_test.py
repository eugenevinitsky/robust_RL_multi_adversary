from gym import Env
from gym.spaces import Box, Discrete

import numpy as np

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

class DummyEnv(Env):

    def __init__(self):
        super().__init__()
        self.num_steps = 0

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(20, ))

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=(2, ))

    def step(self, action):
        # print('step num {}'.format(self.num_steps))
        self.num_steps += 1
        done = False
        if self.num_steps > 50:
            done = True
        return np.zeros(self.observation_space.shape), 1, done, {}

    def reset(self):
        self.num_steps = 0
        return np.zeros(self.observation_space.shape)

def env_creator(env_config):
    return DummyEnv()

if __name__=='__main__':
    ray.init(object_store_memory=int(5e8), redis_max_memory=int(5e8), memory=int(5e8), num_cpus=8)

    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    register_env('DummyEnv', env_creator)

    config['env'] = 'DummyEnv'
    config['use_gae'] = tune.grid_search([False, True])
    config['batch_mode'] = tune.grid_search(['complete_episodes', 'truncate_episodes'])
    config['num_workers'] = 1
    config['model']['fcnet_hiddens'] = [5,5]
    # config['lr'] = tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6])
    config['train_batch_size'] = 50
    config['sample_batch_size'] = 50
    config['sgd_minibatch_size'] = 10
    config['num_sgd_iter'] = 100
    register_env('DummyEnv', env_creator)

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    exp_dict = {
        'name': 'Test_tf1.15.0',
        'run_or_experiment': alg_run,
        'checkpoint_freq': 10,
        'trial_name_creator': trial_str_creator,
        'stop': {
            'training_iteration': 10000
        },
        'config': config,
    }

    run_tune(**exp_dict, queue_trials=False)

