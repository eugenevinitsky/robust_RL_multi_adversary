import errno
from datetime import datetime
import os
import subprocess
import sys

import pytz
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from algorithms.custom_ppo import KLPPOTrainer, CustomPPOPolicy, DEFAULT_CONFIG as CUSTOM_DEFAULT_CONFIG

from utils.parsers import init_parser, env_parser, ray_parser, ma_env_parser
from utils.trajectory_env_creator import ma_env_creator, construct_config

from models.recurrent_tf_model_v2 import LSTM


def setup_ma_config(config):
    env = ma_env_creator(config['env_config'])
    policies_to_train = ['robot1', 'robot2', 'robot3', 'robot4']

    robots_config = {"model": {'fcnet_hiddens': [32, 32], 'use_lstm': False}}
    policy_graphs = {}
    if config['env_config']['kl_diff_training']:
        policy_graphs.update({policies_to_train[i]: (CustomPPOPolicy, env.observation_space,
                                                env.action_space, robots_config) for i in range(4)})
    else:
        policy_graphs.update({policies_to_train[i]: (PPOTFPolicy, env.observation_space,
                                                env.action_space, robots_config) for i in range(4)})

    def policy_mapping_fn(agent_id):
        return agent_id

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
    parser = ma_env_parser(parser)
    parser.add_argument('--custom_ppo', action='store_true', default=False, help='If true, we use the PPO with a KL penalty')

    args = parser.parse_args(args)

    alg_run = 'PPO'


    # Universal hyperparams
    if args.custom_ppo:
        config = CUSTOM_DEFAULT_CONFIG
    else:
        config = DEFAULT_CONFIG
    config['gamma'] = 0.95
    config["batch_mode"] = "complete_episodes"
    config['train_batch_size'] = args.train_batch_size
    config['vf_clip_param'] = 10.0
    config['lambda'] = 0.1
    config['lr'] = 5e-3
    config['sgd_minibatch_size'] = 64
    config['num_sgd_iter'] = 10

    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()
    config['env_config'] = construct_config(env_params, policy_params, args)

    if args.custom_ppo:
        config['kl_diff_weight'] = args.kl_diff_weight
        config['kl_diff_target'] = args.kl_diff_target
        config['kl_diff_clip'] = 5.0
        config['env_config']['kl_diff_training'] = True
    else:
        config['env_config']['kl_diff_training'] = False

    config['env_config']['run'] = alg_run

    if args.grid_search:
        config['vf_loss_coeff'] = tune.grid_search([1e-4, 1e-3])

    config['env'] = 'MATrajectoryEnv'
    register_env('MATrajectoryEnv', ma_env_creator)

    setup_ma_config(config)

    # config["eager_tracing"] = True
    # The custom PPO code flips out here due to a bug in RLlib with eager tracing.
    # Or, at least I think that's what is happening.
    if not args.custom_ppo:
        config["eager"] = True
        config["eager_tracing"] = True

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    if args.custom_ppo:
        trainer = KLPPOTrainer
    else:
        trainer = PPOTrainer
    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': trainer,
        'trial_name_creator': trial_str_creator,
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }
    return exp_dict, args


if __name__ == "__main__":

    exp_dict, args = setup_exps(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/trajectory/' \
                + date + '/' + args.exp_title
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init(local_mode=False)

    run_tune(**exp_dict, queue_trials=False)
