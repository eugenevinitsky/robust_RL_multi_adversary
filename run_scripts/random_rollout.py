"""Rollout the env with random actions"""

import argparse
import configparser
import os
import sys

import numpy as np

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot
from utils.parsers import env_parser, init_parser

def run(passed_config):
    config_path = passed_config['env_config']
    temp_config = configparser.RawConfigParser()
    temp_config.read_string(config_path)

    robot = Robot(temp_config, 'robot')

    env = CrowdSimEnv(temp_config, robot)
    # additional configuration
    env.show_images = passed_config['show_images']
    env.train_on_images = passed_config['train_on_images']

    # configure policy
    policy_config = configparser.RawConfigParser()
    policy_config.read_string(passed_config['policy_config'])
    policy = policy_factory[passed_config['policy']](policy_config)
    if not policy.trainable:
        sys.exit('Policy has to be trainable')
    if passed_config["policy_config"] is None:
        sys.exit('Policy config has to be specified for a trainable network')

    robot.set_policy(policy)

    ob = env.reset()
    total_rew = 0
    for i in range(100):
        ob, rew, done, info = env.step(np.random.rand(2))
        total_rew += rew
    print('The total reward is {}'.format(total_rew))

def main():

    parser = init_parser()
    args = env_parser(parser).parse_args()
    with open(args.env_config, 'r') as file:
        env_config = file.read()

    with open(args.policy_config, 'r') as file:
        policy_config = file.read()

    passed_config = {'env_config': env_config, 'policy_config': policy_config,
                     'policy': args.policy, 'show_images': args.show_images, 'train_on_images': args.train_on_images}
    run(passed_config)


if __name__ == "__main__":
    main()    
