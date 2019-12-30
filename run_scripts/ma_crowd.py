import errno
from datetime import datetime
import random
import os
import subprocess
import sys

import pytz
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer

import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

# from algorithms.custom_ppo import KLPPOTrainer, CustomPPOPolicy
from visualize.visualize_adversaries import visualize_adversaries
from visualize.transfer_test import run_transfer_tests
from utils.env_creator import ma_env_creator, construct_config

from utils.parsers import init_parser, env_parser, ray_parser, ma_env_parser
from utils.rllib_utils import get_config_from_path

from ray.rllib.models.catalog import MODEL_DEFAULTS
from models.conv_lstm import ConvLSTM


def setup_ma_config(config):
    env = ma_env_creator(config['env_config'])
    policies_to_train = ['robot']

    num_adversaries = config['env_config']['num_adversaries']
    adv_policies = ['adversary' + str(i) for i in range(num_adversaries)]
    adversary_config = {}#{"model": {'fcnet_hiddens': [32, 32], 'use_lstm': False}}
    # TODO(@evinitsky) put this back
    # policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
    #                                              env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    # policy_graphs.update({adv_policies[i]: (PPOTFPolicy, env.adv_observation_space,
    #                                         env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    if config['env_config']['run'] == 'DDPG':
        policy_graphs = {'robot': (DDPGTFPolicy, env.observation_space, env.action_space, {})}
        policy_graphs.update({adv_policies[i]: (DDPGTFPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    elif config['env_config']['run'] == 'PPO':
        policy_graphs = {'robot': (PPOTFPolicy, env.observation_space, env.action_space, {})}
        policy_graphs.update({adv_policies[i]: (PPOTFPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    else:
        sys.exit('How did you get here friend? Was there no error catching before this?')

    policies_to_train += adv_policies

    def policy_mapping_fn(agent_id):
        return agent_id

    # def policy_mapping_fn(agent_id):
    #     if agent_id == 'robot':
    #         return agent_id
    #     if agent_id.startswith('adversary'):
    #         return random.choice(adv_policies)

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
    args = parser.parse_args(args)

    alg_run = args.algorithm

    if args.algorithm == 'PPO':
        trainer = PPOTrainer
        config = ppo.DEFAULT_CONFIG.copy()
        config["sgd_minibatch_size"] = 500
        config["num_sgd_iter"] = 10
        config['train_batch_size'] = args.train_batch_size
        if args.grid_search:
            config["lr"] = tune.grid_search([5e-4, 5e-5])
        config['num_workers'] = args.num_cpus
    elif args.algorithm == 'DDPG':
        trainer = DDPGTrainer
        config = ddpg.DEFAULT_CONFIG.copy()
        if args.grid_search:
            config["critic_lr"] = tune.grid_search([1e-4, 1e-3, 1e-2])
            config["actor_lr"] = tune.grid_search([1e-4, 1e-3, 1e-2])
            config["exploration_should_anneal"] = tune.grid_search([True, False])
    else:
        sys.exit('Only PPO and DDPG algorithms are currently supported. Pick a different algorithm bruv.')

    # Universal hyperparams
    config['gamma'] = 0.99
    config["batch_mode"] = "complete_episodes"
    # config['num_adversaries'] = args.num_adv
    # TODO(@evinitsky) put this back
    # config['kl_diff_weight'] = args.kl_diff_weight

    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()
    config['env_config'] = construct_config(env_params, policy_params, args)

    config['env_config']['perturb_state'] = args.perturb_state
    config['env_config']['perturb_actions'] = args.perturb_actions
    config['env_config']['num_adversaries'] = args.num_adv
    config['env_config']['run'] = alg_run

    if not args.perturb_state and not args.perturb_actions:
        sys.exit('You need to select at least one of perturb actions or perturb state')

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
        config['model']['lstm_cell_size'] = 128
        config['model']['custom_options']['fcnet_hiddens'] = [[32, 32], []]
        # If this is true we concatenate the actions onto the network post-convolution
        config['model']['custom_options']['use_prev_action'] = True
        config['model']['conv_filters'] = conv_filters
        config['model']['custom_model'] = "rnn"
        config['vf_share_layers'] = True
    else:
        config['model']['fcnet_hiddens'] = [64, 64]
        config['model']['use_lstm'] = True
        config['model']['lstm_use_prev_action_reward'] = True
        config['model']['lstm_cell_size'] = 128
        if args.algorithm == 'PPO':
            config['vf_share_layers'] = True
            if args.grid_search:
                config['vf_loss_coeff'] = tune.grid_search([1e-4, 1e-3])

    config['env'] = 'MultiAgentCrowdSimEnv'
    register_env('MultiAgentCrowdSimEnv', ma_env_creator)

    setup_ma_config(config)

    # add the callbacks
    config["callbacks"] = {"on_train_result": on_train_result,
                           "on_episode_end": on_episode_end}

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    exp_dict = {
        'name': args.exp_title,
        # TODO (@evinitsky) put this back
        # 'run_or_experiment': KLPPOTrainer,
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


def on_train_result(info):
    """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
    result = info["result"]
    robot_reward = result['policy_reward_mean']['robot']
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.update_mean_rew(robot_reward)))

    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.update_adversary_range()))

    # TODO(should we do this every episode or every training iteration)?
    # trainer.workers.foreach_worker(
    #     lambda ev: ev.foreach_env(
    #         lambda env: env.select_new_adversary()))


def on_episode_end(info):
    """Select the currently active adversary"""

    # store info about how many adversaries there are
    env = info["env"].envs[0]
    episode = info["episode"]
    episode.custom_metrics["num_active_adversaries"] = env.adversary_range

    if env.prediction_reward and env.adversary_range > 1:
        episode.custom_metrics["predict_frac"] = env.num_correct_predict / episode.length

    # select a new adversary every episode. Currently disabled.
    if env.adversary_range > 0:
        env.curr_adversary = np.random.randint(low=0, high=env.adversary_range)
    else:
        env.curr_adversary = 0


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
                # TODO(@evinitsky) this will break for state adversaries
                visualize_adversaries(config, checkpoint_path, 20, 500, output_path)
                p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path, "s3://sim2real/transfer_results/{}/{}/{}".format(date, args.exp_title, tune_name)).split(' '))
                p1.wait()