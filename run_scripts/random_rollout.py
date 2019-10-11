"""Rollout the env with random actions. Useful for testing features even when you don't have a trained
policy. """

import argparse
import configparser
import os

import numpy as np

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory

parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('--env_config', type=str, default=os.path.abspath('../configs/env.config'))
parser.add_argument('--policy_config', type=str, default=os.path.abspath('../configs/policy.config'))
parser.add_argument('--policy', type=str, default='cadrl')
parser.add_argument('--train_config', type=str, default='../configs/train.config')
parser.add_argument("--show_images", action="store_true", help="Whether to display the observations")
parser.add_argument('--train_on_images', action='store_true', help='Whether to train on images')

# Arguments for transfer tests
parser.add_argument('--add_friction', action='store_true', help='If true, there is `friction` in the '
                                                                'dynamics')
parser.add_argument('--change_colors_mode', type=str, default='no_change',
                    help='If mode `every_step`, the colors will be swapped '
                         'at each step. If mode `first_step` the colors will'
                         'be swapped only once')
parser.add_argument('--kinematics', type='str', default='holonomic',
                    help='Type of action space. Options are holonomic and unicycle')

args = parser.parse_args()

passed_config = {'config_path': args.env_config, 'policy_config': args.policy_config,
                 'policy': args.policy, 'show_images': args.show_images,
                 'train_on_images': args.train_on_images}

config_path = passed_config['config_path']
temp_config = configparser.RawConfigParser()
temp_config.read(config_path)
if args.add_friction:
    friction = 'true'
else:
    friction = 'false'
temp_config.set('transfer', 'friction', friction)
env = CrowdSimEnv(temp_config, train_on_images=passed_config['train_on_images'],
                  show_images=passed_config['show_images'])

# additional configuration
env.change_colors_mode = args.change_colors_mode

# configure policy
policy_config = configparser.RawConfigParser()
policy_config.read(passed_config['policy_config'])
policy = policy_factory[passed_config['policy']](policy_config)
# Update the transfer params in the policy
policy.set('action_space', 'kinematics', args.kinematics)
if not policy.trainable:
    parser.error('Policy has to be trainable')
if args.policy_config is None:
    parser.error('Policy config has to be specified for a trainable network')

env.robot.set_policy(policy)

ob = env.reset()
for i in range(100):
    ob, rew, done, info = env.step(0.2 * np.random.normal(size=2))
