import argparse
import configparser
from datetime import datetime
import os
import sys

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot
from utils.parsers import init_parser, env_parser, ray_parser


def setup_exps():
    parser = init_parser()
    parser = env_parser(parser)
    parser = ray_parser(parser)
    args = parser.parse_args()

    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = args.num_cpus
    config['gamma'] = 0.99
    config['train_batch_size'] = 10000

    config['env_config']['replay_params'] = vars(args)
    config['env_config']['run'] = alg_run

    # pick out the right model
    if args.train_on_images:
        # register the custom model
        conv_filters = [
            [32, [3, 3], 2],
            [32, [3, 3], 2],
        ]
        config['model'] = {'conv_activation': 'relu', 'use_lstm': True, "lstm_use_prev_action_reward": True,
                           'lstm_cell_size': 128, 'conv_filters': conv_filters}
        config['vf_share_layers'] = True
        config['train_batch_size'] = 500  # TODO(@evinitsky) change this it's just for testing
    else:
        config['model'] = {'use_lstm': True, "lstm_use_prev_action_reward": True, 'lstm_cell_size': 128}
        config['vf_share_layers'] = True
        config['vf_loss_coeff'] = 1e-4

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()
    s3_string = 's3://sim2real/' \
                + datetime.now().strftime('%m-%d-%Y') + '/' + args.exp_title
    config['env'] = 'CrowdSim'
    register_env('CrowdSim', env_creator)

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': alg_run,
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    return alg_run, config, exp_dict, args


def env_creator(passed_config):
    config_path = passed_config['env_config']
    temp_config = configparser.RawConfigParser()
    temp_config.read_string(config_path)
    env = CrowdSimEnv()
    env.configure(temp_config)
    # additional configuration
    env.show_images = passed_config['show_images']
    env.train_on_images = passed_config['train_on_images']

    robot = Robot(temp_config, 'robot')
    env.set_robot(robot)

    # configure policy
    policy_config = configparser.RawConfigParser()
    policy_config.read_string(passed_config['policy_config'])
    policy = policy_factory[passed_config['policy']](policy_config)
    if not policy.trainable:
        sys.exit('Policy has to be trainable')
    if passed_config['policy_config'] is None:
        sys.exit('Policy config has to be specified for a trainable network')

    robot.set_policy(policy)
    return env


if __name__=="__main__":

    alg_run, config, exp_dict, args = setup_exps()

    with open(args.env_config, 'r') as file:
        env_config = file.read()

    with open(args.policy_config, 'r') as file:
        policy_config = file.read()

    # save the relevant params for replay
    exp_dict['config']['env_config'] = {'policy': args.policy, 'show_images': args.show_images,
                                        'train_on_images': args.train_on_images,
                                        'env_config': env_config, 'policy_config': policy_config}

    run_tune(**exp_dict, queue_trials=False)

