import errno
from datetime import datetime
import random
import os
import subprocess
import sys

import pytz
import numpy as np
import ray
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from visualize.transfer_test import run_transfer_tests
from utils.env_creator import ma_env_creator, construct_config

from utils.parsers import init_parser, env_parser, ray_parser
from utils.rllib_utils import get_config_from_path

from ray.rllib.models.catalog import MODEL_DEFAULTS


def setup_ma_config(config):
    env = ma_env_creator(config['env_config'])
    policies_to_train = ['robot']

    adv_policies = ['adversary']
    adversary_config = {"model": {'fcnet_hiddens': [32, 32], 'use_lstm': False}}
    policy_graphs = {'robot': (None, env.observation_space, env.action_space, {})}
    policy_graphs.update({'adversary': (None, env.adv_observation_space,
                                                 env.adv_action_space, adversary_config)})
    policies_to_train += adv_policies

    def policy_mapping_fn(agent_id):
        if 'human' in agent_id:
            return 'adversary'
        else:
            return 'robot'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': policy_mapping_fn,
            'policies_to_train': policies_to_train
        }
    })


def setup_exps(args):
    parser = init_parser()
    parser = env_parser(parser)
    parser = ray_parser(parser)
    args = parser.parse_args(args)

    alg_run = 'PPO'

    # Universal hyperparams
    config = DEFAULT_CONFIG
    config['gamma'] = 0.99
    config["batch_mode"] = "complete_episodes"
    config['train_batch_size'] = args.train_batch_size

    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()
    config['env_config'] = construct_config(env_params, policy_params, args)

    config['env_config']['num_adversaries'] = args.num_adv
    config['env_config']['run'] = alg_run

    # pick out the right model
    if args.train_on_images:
        # register the custom model

        conv_filters = [
            [32, [3, 3], 2],
            [32, [3, 3], 2],
        ]
        config['model'] = MODEL_DEFAULTS.copy()
        
        config['model']['conv_activation'] = 'relu'
        config['model']['lstm_cell_size'] = 128
        config['model']['custom_options']['fcnet_hiddens'] = [[32, 32], []]
        # If this is true we concatenate the actions onto the network post-convolution
        config['model']['custom_options']['use_prev_action'] = True
        config['model']['conv_filters'] = conv_filters
        config['model']['custom_model'] = "rnn"
        config['vf_share_layers'] = True
    else:
        config['model']['custom_options']['fcnet_hiddens'] = [64, 64]
        config['model']['use_lstm'] = False
        config['model']['lstm_use_prev_action_reward'] = False
        config['model']['lstm_cell_size'] = 128
        config['vf_share_layers'] = True
        if args.grid_search:
            config['vf_loss_coeff'] = tune.grid_search([1e-4, 1e-3])

    config['env'] = 'MultiAgentCrowdSimEnv'
    register_env('MultiAgentCrowdSimEnv', ma_env_creator)

    setup_ma_config(config)

    # You don't have to do this, but it does make replay a little cleaner
    config["eager"] = True
    config["eager_tracing"] = True

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': "PPO",
        'trial_name_creator': trial_str_creator,
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }
    return exp_dict, args


if __name__=="__main__":

    exp_dict, args = setup_exps(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/' \
                + date + '/' + args.exp_title
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()

    run_tune(**exp_dict, queue_trials=False)

    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results'), date), args.exp_title)
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
            if "checkpoint_{}".format(args.num_iters) in dirpath:
                # grab the experiment name
                folder = os.path.dirname(dirpath)
                tune_name = folder.split("/")[-1]
                outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.expanduser(os.path.join(outer_folder, "visualize/transfer_test.py"))
                config, checkpoint_path = get_config_from_path(folder, str(args.num_iters))

                run_transfer_tests(config, checkpoint_path, 500, args.exp_title, output_path, save_trajectory=False)
                p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path, "s3://sim2real/transfer_results/{}/{}/{}".format(date, args.exp_title, tune_name)).split(' '))
                p1.wait()