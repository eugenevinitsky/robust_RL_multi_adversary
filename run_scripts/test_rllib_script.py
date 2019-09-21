import argparse
import configparser
from datetime import datetime
import os

import gym
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import run
from ray.tune.registry import register_env

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot


def setup_exps(args):
    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = args.num_cpus
    return alg_run, config


def env_creator(passed_config):
    config_path = passed_config['config_path']
    temp_config = configparser.RawConfigParser()
    temp_config.read(config_path)
    env = CrowdSimEnv()
    env.configure(temp_config)
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
    return env


if __name__=="__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default=os.path.abspath('../configs/env.config'))
    parser.add_argument('--policy_config', type=str, default=os.path.abspath('../configs/policy.config'))
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--train_config', type=str, default='../configs/train.config')
    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('exp_title', type=str, help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument("--num_iters", type=int, default=350)
    parser.add_argument("--checkpoint_freq", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    env = env_creator({'config_path': args.env_config})
    env.reset(phase='train')
    register_env("CrowdSim", env_creator)

    alg_run, config = setup_exps(args)
    config['env_config'] = {'config_path': args.env_config}
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()
    s3_string = "s3://eugene.experiments/sim2real/" \
                + datetime.now().strftime("%m-%d-%Y") + '/' + args.exp_title
    config['env'] = "CrowdSim"
    exp_dict = {
            'name': args.exp_title,
            'run_or_experiment': 'PPO',
            'checkpoint_freq': args.checkpoint_freq,
            'stop': {
                'training_iteration': args.num_iters
            },
            'config': config,
            'num_samples': args.num_samples,
        }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    run(**exp_dict, queue_trials=False)
