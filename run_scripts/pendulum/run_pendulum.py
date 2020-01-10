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

from visualize.pendulum.transfer_tests import run_transfer_tests
from visualize.pendulum.visualize_adversaries import visualize_adversaries
from utils.parsers import init_parser, ray_parser, ma_env_parser
from utils.pendulum_env_creator import pendulum_env_creator
from utils.rllib_utils import get_config_from_path

from models.recurrent_tf_model_v2 import LSTM


def setup_ma_config(config):
    env = pendulum_env_creator(config['env_config'])
    policies_to_train = ['pendulum']

    num_adversaries = config['env_config']['num_adversaries']
    if num_adversaries == 0:
        return
    adv_policies = ['adversary' + str(i) for i in range(num_adversaries)]
    adversary_config = {"model": {'fcnet_hiddens': [32, 32], 'use_lstm': False}}
    policy_graphs = {'pendulum': (PPOTFPolicy, env.observation_space, env.action_space, {})}
    if config['env_config']['kl_diff_training']:
        policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    else:
        policy_graphs.update({adv_policies[i]: (PPOTFPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    # TODO(@evinitsky) put this back

    policies_to_train += adv_policies

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
    parser.add_argument('--num_adv_per_strength', type=int, default=1,
                        help='This value sets how many adversaries exist per strength level. The adversary strengths'
                             'are set by evenly gridding from 0 to --adv_strength. At each of these levels,'
                             'we will have `num_adv_per_strength` adversaries')
    parser.add_argument('--adv_incr_freq', type=int, default=10,
                        help='How many iterations we should stably maintain a score of `turn_on_score` before '
                             'increasing the number of adversaries')
    parser.add_argument('--curriculum', action='store_true', default=False,
                        help='If true, we gradually increase the number of adversaries by `num_adv_per_strength`'
                             'for every `adv_incr_freq` that we maintain a score above `goal_score`')
    parser.add_argument('--goal_score', type=float, default=-200,
                        help='If curriculum is on, we increase the number of adversaries by `num_adv_per_strength`'
                             'for every `adv_incr_freq` that we maintain a score above this value')

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

    if args.custom_ppo:
        config['num_adversaries'] = args.num_adv * args.num_adv_per_strength
        config['kl_diff_weight'] = args.kl_diff_weight
        config['kl_diff_target'] = args.kl_diff_target
        config['kl_diff_clip'] = 5.0
        config['env_config']['kl_diff_training'] = True
    else:
        config['env_config']['kl_diff_training'] = False

    config['env_config']['num_adversaries'] = args.num_adv * args.num_adv_per_strength
    config['env_config']['adversary_strength'] = args.adv_strength
    config['env_config']['num_adv_per_strength'] = args.num_adv_per_strength
    config['env_config']['adv_incr_freq'] = args.adv_incr_freq
    config['env_config']['curriculum'] = args.curriculum
    config['env_config']['goal_score'] = args.goal_score

    config['env_config']['run'] = alg_run

    ModelCatalog.register_custom_model("rnn", LSTM)
    config['model']['fcnet_hiddens'] = [64, 64]
    # TODO(@evinitsky) turn this on
    config['model']['use_lstm'] = False
    # config['model']['custom_model'] = "rnn"
    config['model']['lstm_use_prev_action_reward'] = False
    config['model']['lstm_cell_size'] = 128
    if args.grid_search:
        config['vf_loss_coeff'] = tune.grid_search([1e-4, 1e-3])

    config['env'] = 'MAPendulumEnv'
    register_env('MAPendulumEnv', pendulum_env_creator)

    setup_ma_config(config)

    # add the callbacks
    config["callbacks"] = {"on_train_result": on_train_result,
                           "on_episode_end": on_episode_end}

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


def on_train_result(info):
    """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
    result = info["result"]
    pendulum_reward = result['policy_reward_mean']['pendulum']
    trainer = info["trainer"]

    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.update_curriculum(pendulum_reward)))


def on_episode_end(info):
    """Select the currently active adversary"""

    # store info about how many adversaries there are
    if hasattr(info["env"], 'envs'):
        env = info["env"].envs[0]
        env.select_new_adversary()

        episode = info['episode']
        episode.custom_metrics["num_active_advs"] = env.adversary_range

    elif hasattr(info["env"], 'vector_env'):
        envs = info["env"].vector_env.envs
        for env in envs:
            env.select_new_adversary()


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
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init(local_mode=False)

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

                run_transfer_tests(config, checkpoint_path, 100, args.exp_title, output_path)
                if args.num_adv > 0:

                    visualize_adversaries(config, checkpoint_path, 10, 100, output_path)
                    if args.use_s3:
                        p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                         "s3://sim2real/transfer_results/pendulum/{}/{}/{}".format(date,
                                                                                                                          args.exp_title,
                                                                                                                          tune_name)).split(
                            ' '))
                        p1.wait()