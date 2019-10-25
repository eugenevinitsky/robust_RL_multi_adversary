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
    config_path = passed_config['env_params']
    temp_config = configparser.RawConfigParser()
    temp_config.read_string(config_path)

    robot = Robot(temp_config, 'robot')

    env = CrowdSimEnv(temp_config, robot)
    # additional configuration
    env.show_images = passed_config['show_images']

    # configure policy
    policy_params = configparser.RawConfigParser()
    policy_params.read_string(passed_config['policy_params'])
    policy = policy_factory[passed_config['policy']](policy_params)
    if not policy.trainable:
        sys.exit('Policy has to be trainable')
    if passed_config["policy_params"] is None:
        sys.exit('Policy config has to be specified for a trainable network')

    robot.set_policy(policy)

    ob = env.reset()
    total_rew = 0
    for i in range(100):
        ob, rew, done, info = env.step(np.random.rand(2))
        total_rew += rew
    print('The total reward is {}'.format(total_rew))

def setup_random():
    parser = init_parser()
    args = env_parser(parser).parse_args()
    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()

    passed_config = {'env_params': env_params, 'policy_params': policy_params,
                     'policy': args.policy, 'show_images': args.show_images}

    return passed_config

def main():
    config = setup_random()

    run(config)


if __name__ == "__main__":
    main()    
