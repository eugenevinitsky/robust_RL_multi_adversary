"""Rollout the env with random actions"""

import argparse
import configparser
import os

import numpy as np

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    script_path = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('--env_config', type=str, default=os.path.abspath(os.path.join(script_path,'../configs/env.config')))
    parser.add_argument('--policy_config', type=str, default=os.path.abspath(os.path.join(script_path,'../configs/policy.config')))
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument("--show_images", action="store_true", default=False, help="Whether to display the observations")
    parser.add_argument('--train_on_images', action='store_true', default=False, help='Whether to train on images')

    args = parser.parse_args()

    with open(args.env_config, 'r') as file:
        env_config = file.read()

    with open(args.policy_config, 'r') as file:
        policy_config = file.read()

    passed_config = {'env_config': env_config, 'policy_config': policy_config,
                                'policy': args.policy, 'show_images': args.show_images, 'train_on_images': args.train_on_images}

    config_path = passed_config['env_config']
    temp_config = configparser.RawConfigParser()
    temp_config.read_string(config_path)
    env = CrowdSimEnv(temp_config)
    # additional configuration
    env.show_images = passed_config['show_images']
    env.train_on_images = passed_config['train_on_images']

    robot = Robot(temp_config, 'robot')
    env.set_robot(robot)

    # configure policy
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy = policy_factory[args.policy](policy_config)

    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')

    robot.set_policy(policy)

    ob = env.reset()
    total_rew = 0
    for i in range(100):
        ob, rew, done, info = env.step(np.random.rand(2))
        total_rew += rew
    print('The total reward is {}'.format(total_rew))

if __name__ == "__main__":
    main()    
