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

from utils.env_creator import env_creator, construct_config
from utils.parsers import init_parser, env_parser, ray_parser

from ray.rllib.models.catalog import MODEL_DEFAULTS
from models.conv_lstm import ConvLSTM


def setup_exps(args):
    parser = init_parser()
    parser = env_parser(parser)
    parser = ray_parser(parser)
    args = parser.parse_args(args)

    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = args.num_cpus
    config['gamma'] = 0.99
    config['train_batch_size'] = 30000

    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()

    config['env_config'] = construct_config(env_params, policy_params, args)
    config['env_config']['run'] = alg_run

    # pick out the right model
    if args.train_on_images:
        # register the custom model
        ModelCatalog.register_custom_model("rnn", ConvLSTM)

        conv_filters = [
            [32, [3, 3], 2],
            [32, [3, 3], 2],
        ]
        config['model'] = MODEL_DEFAULTS.copy()
        
        config['model']['conv_activation'] = 'relu'
        config['model']['use_lstm'] = True
        config['model']['lstm_use_prev_action_reward'] = True
        config['model']['lstm_cell_size'] = 128
        config['model']['custom_options']['fcnet_hiddens'] = [[32, 32], []]
        config['model']['conv_filters'] = conv_filters
        config['model']['custom_model'] = "rnn"
        
        config['vf_share_layers'] = True
        config['sgd_minibatch_size'] = 600
        config['num_sgd_iter'] = 10
        config['train_batch_size'] = 5000

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

    return exp_dict, args


if __name__=="__main__":

    exp_dict, args = setup_exps(sys.argv[1:])
    
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()

    run_tune(**exp_dict, queue_trials=False)

