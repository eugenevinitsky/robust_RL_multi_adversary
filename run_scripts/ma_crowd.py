import argparse
import configparser
from datetime import datetime
import os

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune import run
from ray.tune.registry import register_env

from gym.spaces import Box

from envs.crowd_env import PerturbObsEnv
from envs.policy.policy_factory import policy_factory


def setup_exps(args):
    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = args.num_cpus
    return alg_run, config


def env_creator(passed_config):
    config_path = passed_config['config_path']
    temp_config = configparser.RawConfigParser()
    temp_config.read(config_path)
    env = PerturbObsEnv(temp_config, train_on_images=passed_config['train_on_images'],
                        show_images=passed_config['show_images'])

    # configure policy
    policy_config = configparser.RawConfigParser()
    policy_config.read(passed_config['policy_config'])
    policy = policy_factory[passed_config['policy']](policy_config)
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if args.policy_config is None:
        parser.error('Policy config has to be specified for a trainable network')

    env.robot.set_policy(policy)
    return env

def setup_ma_config(config):
        # Could use suggestions on how we want to store / pass 
        # these values (KJ)
        config_path = config['env_config']['config_path']
        temp_config = configparser.RawConfigParser()
        temp_config.read(config_path)
        num_humans = temp_config.getint('sim', 'human_num')

        temp_env = env_creator(config)

        adv_action_space = Box(low=-1.0, high=1.0, shape=(num_humans * 5, ))

        policies_to_train = ['robot', 'adversary']

        policy_graphs = {'robot': (None, temp_env.observation_space, temp_env.action_space, {}),
                        'adversary': (None, temp_env.observation_space, adv_action_space, {})}

        def policy_mapping_fn(agent_id):
            if agent_id != 'robot':
                return 'adversary'
            else:
                return agent_id
        
        policy_ids = list(policy_graphs.keys())

        config.update({
            'multiagent': {
                'policy_graphs': policy_graphs,
                'policy_mapping_fn': tune.function(policy_mapping_fn),
                'policies_to_train': policies_to_train
            }
        })


if __name__=="__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default=os.path.abspath(os.path.join(script_path,'../configs/env.config')))
    parser.add_argument('--policy_config', type=str, default=os.path.abspath(os.path.join(script_path,'../configs/policy.config')))
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--train_config', type=str, default=os.path.join(script_path,'../configs/train.config'))
    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('exp_title', type=str, help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument('--num_iters', type=int, default=350)
    parser.add_argument('--checkpoint_freq', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)

    # Env configs
    parser.add_argument('--show_images', action='store_true', default=False, help='Whether to display the observations')
    parser.add_argument('--train_on_images', action='store_true', default=False, help='Whether to train on images')

    args = parser.parse_args()

    register_env('CrowdSim', env_creator)

    alg_run, config = setup_exps(args)

    # save the relevant params for replay
    config['env_config'] = {'config_path': args.env_config, 'policy_config': args.policy_config,
                            'policy': args.policy, 'show_images': args.show_images, 'train_on_images': args.train_on_images}
    config['env_config']['replay_params'] = vars(args)
    config['env_config']['run'] = alg_run

    # pick out the right model
    if args.train_on_images:
        # register the custom model
        conv_filters = [
                [32, [3, 3], 2],
                [32, [3, 3], 2],
            ]
        config['model'] = {'conv_activation': 'relu', 'use_lstm': True,
                           'lstm_cell_size': 128, 'conv_filters': conv_filters}
        config['vf_share_layers'] = True
        config['train_batch_size']: 500  # TODO(@evinitsky) change this it's just for testing

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()
    s3_string = 's3://eugene.experiments/sim2real/' \
                + datetime.now().strftime('%m-%d-%Y') + '/' + args.exp_title
    config['env'] = 'CrowdSim'

    setup_ma_config(config)

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

    run(**exp_dict, queue_trials=False)
