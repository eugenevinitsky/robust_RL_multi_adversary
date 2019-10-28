"""Rollout the env with random actions"""

import argparse
import configparser
import os
import sys

import numpy as np

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot
from utils.env_creator import env_creator, construct_config
from utils.parsers import env_parser, init_parser


def run(passed_config):
    env = env_creator(passed_config)

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

    passed_config = construct_config(env_params, policy_params, args)

    return passed_config

def main():
    config = setup_random()

    run(config)


if __name__ == "__main__":
    main()    
