import errno
from datetime import datetime
from functools import reduce
import random
import os
import subprocess
import sys

import pytz
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

# from algorithms.custom_ppo import KLPPOTrainer, CustomPPOPolicy, DEFAULT_CONFIG

from visualize.pendulum.transfer_tests import run_transfer_tests
from visualize.pendulum.model_adversary_grid import visualize_model_perf
from visualize.pendulum.visualize_adversaries import visualize_adversaries
from utils.parsers import init_parser, ray_parser, ma_env_parser
from utils.pendulum_env_creator import pendulum_env_creator
from utils.rllib_utils import get_config_from_path

from models.recurrent_tf_model_v2 import LSTM


def setup_ma_config(config):
    env = pendulum_env_creator(config['env_config'])

    num_adversaries = config['env_config']['num_adversaries']

    if num_adversaries > 0 and not config['env_config']['model_based']:
        policies_to_train = ['pendulum']
        policy_graphs = {'pendulum': (PPOTFPolicy, env.observation_space, env.action_space, {})}
        adv_policies = ['adversary' + str(i) for i in range(num_adversaries)]
        adversary_config = {"model": {'fcnet_hiddens': [32, 32], 'use_lstm': False}}
        policy_graphs.update({adv_policies[i]: (PPOTFPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    # TODO(@evinitsky) put this back
    # policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
    #                                         env.adv_action_space, adversary_config) for i in range(num_adversaries)})

        policies_to_train += adv_policies
    else:
        return

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
    parser = ray_parser(parser)
    parser = ma_env_parser(parser)
    parser.add_argument('--custom_ppo', action='store_true', default=False, help='If true, we use the PPO with a KL penalty')
    parser.add_argument('--num_adv', type=int, default=5, help='Number of active adversaries in the env')
    parser.add_argument('--adv_strength', type=float, default=0.1, help='Strength of active adversaries in the env')
    parser.add_argument('--model_based', action='store_true', default=False,
                        help='If true, the adversaries are a set of fixed sinusoids instead of being learnt')
    parser.add_argument('--guess_adv', action='store_true', default=False,
                        help='If true, a prediction head is added that the agent uses to guess '
                             'which adversary is currently active')
    parser.add_argument('--guess_next_state', action='store_true', default=False,
                        help='If true, a prediction head is added that the agent uses to guess '
                             'what the next state is going to be')
    args = parser.parse_args(args)

    alg_run = 'PPO'

    # Universal hyperparams
    config = DEFAULT_CONFIG
    config['gamma'] = 0.95
    config["batch_mode"] = "complete_episodes"
    config['train_batch_size'] = args.train_batch_size
    config['vf_clip_param'] = 10.0
    config['lambda'] = 0.1
    config['lr'] = 5e-3
    config['sgd_minibatch_size'] = 64
    config['num_envs_per_worker'] = 10
    config['num_sgd_iter'] = 10

    if args.custom_ppo:
        config['num_adversaries'] = args.num_adv
        config['kl_diff_weight'] = args.kl_diff_weight
        config['kl_diff_target'] = args.kl_diff_target
        config['kl_diff_clip'] = 5.0

    # Options used in every env
    config['env_config']['num_adversaries'] = args.num_adv
    config['env_config']['adversary_strength'] = args.adv_strength
    config['env_config']['model_based'] = args.model_based
    # These next options are only used in the model based env
    config['env_config']['guess_adv'] = args.guess_adv
    config['env_config']['guess_next_state'] = args.guess_next_state

    config['env_config']['run'] = alg_run

    ModelCatalog.register_custom_model("rnn", LSTM)
    config['model']['fcnet_hiddens'] = [64, 64]
    # TODO(@evinitsky) turn this on
    # config['model']['use_lstm'] = True
    # config['model']['custom_model'] = "rnn"
    config['model']['custom_options'] = {'lstm_use_prev_action': False}
    config['model']['lstm_cell_size'] = 128
    config['model']['max_seq_len'] = 20
    if args.grid_search:
        config['vf_loss_coeff'] = tune.grid_search([1e-4, 1e-3])

    config['env'] = 'MAPendulumEnv'
    register_env('MAPendulumEnv', pendulum_env_creator)

    setup_ma_config(config)

    # add the callbacks
    config["callbacks"] = {"on_train_result": on_train_result,
                           "on_episode_end": on_episode_end}

    # config["eager_tracing"] = True
    # config["eager"] = True
    # config["eager_tracing"] = True

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    exp_dict = {
        'name': args.exp_title,
        # 'run_or_experiment': KLPPOTrainer,
        'run_or_experiment': 'PPO',
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
    # pendulum_reward = result['policy_reward_mean']['pendulum']
    trainer = info["trainer"]

    # TODO(should we do this every episode or every training iteration)?
    pass
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.select_new_adversary()))


def on_episode_end(info):
    """Select the currently active adversary"""

    # store info about how many adversaries there are
    if hasattr(info["env"], 'envs'):

        env = info["env"].envs[0]

        prediction = env.num_correct_guesses / env.horizon

        state_list = env.state_error / env.horizon
        # Track the mean error in every element of the state
        state_err_dict = {"state_err_{}".format(i): sum_state for i, sum_state in enumerate(state_list)}

        episode = info["episode"]
        episode.custom_metrics["correct_pred_frac"] = prediction
        for key, val in state_err_dict.items():
            episode.custom_metrics[key] = val

        # info["episode"].custom_metrics.update(state_err_dict)

        env.select_new_adversary()
    elif hasattr(info["env"], 'vector_env'):
        envs = info["env"].vector_env.envs
        prediction_list = [env.num_correct_guesses / env.horizon for env in envs]
        # some of the envs won't actually be used if we are vectorizing so lets just ignore them
        prediction_list = [prediction for env, prediction in zip(envs, prediction_list) if env.step_num != 0]

        state_list = [env.state_error / env.horizon for env in envs]
        sum_states = reduce(lambda x, y: x+y, state_list) / len(envs)
        # Track the mean error in every element of the state
        state_err_dict = {"state_err_{}".format(i): sum_state for i, sum_state in enumerate(sum_states)}

        info["episode"].custom_metrics["correct_pred_frac"] = np.mean(prediction_list)
        for key, val in state_err_dict.items():
            info["episode"].custom_metrics[key] = val
        for env in envs:
            env.select_new_adversary()
    else:
        sys.exit("You aren't recording any custom metrics, something is wrong")


if __name__ == "__main__":

    exp_dict, args = setup_exps(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/pendulum/' \
                + date + '/' + args.exp_title
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init(local_mode=True)

    run_tune(**exp_dict, queue_trials=False)

    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/pendulum'), date), args.exp_title)
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

                if args.num_adv > 0:
                    run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path)

                    if not args.model_based:
                        visualize_adversaries(config, checkpoint_path, 10, 200, output_path)

                    if args.model_based:
                        visualize_model_perf(config, checkpoint_path, 10, 20, output_path)

                    p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                     "s3://sim2real/transfer_results/pendulum/{}/{}/{}".format(date,
                                                                                                                      args.exp_title,
                                                                                                                      tune_name)).split(
                        ' '))
                    p1.wait()