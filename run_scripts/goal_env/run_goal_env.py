from copy import deepcopy
import errno
from datetime import datetime
import os
import subprocess
import sys

import pytz
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as DEFAULT_PPO_CONFIG

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from algorithms.multi_active_ppo import CustomPPOPolicy, CustomPPOTrainer
from algorithms.custom_kl_distribution import LogitsDist
from envs.goal_env import GoalEnv
from visualize.goal_env.visualize_adversaries import visualize_adversaries
# from visualize.pendulum.visualize_adversaries import visualize_adversaries
from utils.pendulum_env_creator import make_create_env
from utils.parsers import init_parser, ray_parser, ma_env_parser
from utils.rllib_utils import get_config_from_path

from models.recurrent_tf_model_v2 import LSTM


def setup_ma_config(config, create_env):
    env = create_env(config['env_config'])
    policies_to_train = ['agent']

    num_adversaries = config['env_config']['num_adv_strengths'] * config['env_config']['advs_per_strength']
    if num_adversaries == 0:
        return
    adv_policies = ['adversary' + str(i) for i in range(num_adversaries)]
    adversary_config = {"model": {'fcnet_hiddens': [64, 64], 'use_lstm': False}}
    if config['env_config']['kl_reward']:
        ModelCatalog.register_custom_action_dist("logits_dist", LogitsDist)
        adversary_config['model']['custom_action_dist'] = "logits_dist"
    # for both of these we need a graph that zeros out agents that weren't active
    if config['env_config']['kl_reward'] or config['env_config']['l2_reward']:
        policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, {})}
        policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in
                              range(num_adversaries)})
    else:
        policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, {})}
        policy_graphs.update({adv_policies[i]: (PPOTFPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in
                              range(num_adversaries)})

    # TODO(@evinitsky) put this back
    # policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
    #                                         env.adv_action_space, adversary_config) for i in range(num_adversaries)})

    print("========= Policy Graphs ==========")
    print(policy_graphs)

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
    print({'multiagent': {
        'policies': policy_graphs,
        'policy_mapping_fn': policy_mapping_fn,
        'policies_to_train': policies_to_train
    }})


def setup_exps(args):
    parser = init_parser()
    parser = ray_parser(parser)
    parser = ma_env_parser(parser)
    parser.add_argument('--env_name', default='pendulum', const='pendulum', nargs='?',
                        choices=['pendulum', 'hopper', 'cheetah'])
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--algorithm', default='PPO', type=str, help='Options are PPO')
    parser.add_argument('--num_adv_strengths', type=int, default=1, help='Number of adversary strength ranges. '
                                                                         'Multiply this by `advs_per_strength` to get the total number of adversaries'
                                                                         'Default - retrain lerrel, single agent')
    parser.add_argument('--advs_per_strength', type=int, default=1,
                        help='How many adversaries exist at each strength level')
    parser.add_argument('--reward_range', action='store_true', default=False,
                        help='If true, the adversaries try to get agents to goals evenly spaced between `low_reward`'
                             'and `high_reward')
    parser.add_argument('--low_reward', type=float, default=-60.0, help='The lower range that adversaries try'
                                                                      'to push you to')
    parser.add_argument('--high_reward', type=float, default=-1.0, help='The upper range that adversaries try'
                                                                          'to push you to')
    parser.add_argument('--l2_reward', action='store_true', default=False,
                        help='If true, each adversary gets a reward for being close to the adversaries. This '
                             'is NOT a supervised loss')
    parser.add_argument('--l2_reward_coeff', type=float, default=2.0,
                        help='Scaling on the l2_reward')
    parser.add_argument('--l2_in_tranche', action='store_true', default=False,
                        help='If this is true, you only compare l2 values for adversaries that have the same reward '
                             'goal as you ')
    # Unused right now
    parser.add_argument('--kl_reward', action='store_true', default=False,
                        help='If true, each adversary gets a reward for being close to the adversaries in '
                             'KL space.')
    parser.add_argument('--kl_reward_coeff', type=float, default=1.0,
                        help='Scaling on the kl_reward')
    args = parser.parse_args(args)

    if args.reward_range and args.num_adv_strengths * args.advs_per_strength <= 0:
        sys.exit('must specify number of strength levels, number of adversaries when using reward range')

    alg_run = args.algorithm

    if args.algorithm == 'PPO':
        config = deepcopy(DEFAULT_PPO_CONFIG)
        config['train_batch_size'] = args.train_batch_size
        config['gamma'] = 0.95
        if args.grid_search:
            config['lambda'] = tune.grid_search([0.5, 0.9, 1.0])
            config['lr'] = tune.grid_search([5e-5, 5e-4, 5e-3])
        else:
            config['lambda'] = 0.97
            config['lr'] = 5e-4
        config['sgd_minibatch_size'] = 64 * max(int(args.train_batch_size / 1e4), 1)
        if args.use_lstm:
            config['sgd_minibatch_size'] *= 5
        config['num_sgd_iter'] = 10
        config['observation_filter'] = 'NoFilter'

    if config['observation_filter'] == 'MeanStdFilter' and args.l2_reward:
        sys.exit('Mean std filter MUST be off if using the l2 reward')

    # Universal hyperparams
    config['num_workers'] = args.num_cpus
    config['seed'] = 0

    config['env_config']['horizon'] = args.horizon
    config['env_config']['num_adv_strengths'] = args.num_adv_strengths
    config['env_config']['advs_per_strength'] = args.advs_per_strength
    config['env_config']['reward_range'] = args.reward_range
    config['env_config']['low_reward'] = args.low_reward
    config['env_config']['high_reward'] = args.high_reward
    config['env_config']['l2_reward'] = args.l2_reward
    config['env_config']['kl_reward'] = args.kl_reward
    config['env_config']['l2_reward_coeff'] = args.l2_reward_coeff
    config['env_config']['kl_reward_coeff'] = args.kl_reward_coeff
    config['env_config']['l2_in_tranche'] = args.l2_in_tranche

    config['env_config']['run'] = alg_run

    ModelCatalog.register_custom_model("rnn", LSTM)
    config['model']['fcnet_hiddens'] = [64, 64]
    # TODO(@evinitsky) turn this on
    if args.use_lstm:
        config['model']['fcnet_hiddens'] = [64]
        config['model']['use_lstm'] = False
        config['model']['lstm_use_prev_action_reward'] = True
        config['model']['lstm_cell_size'] = 64

    env_name = "GoalEnv"
    create_env_fn = make_create_env(GoalEnv)

    config['env'] = env_name
    register_env(env_name, create_env_fn)

    setup_ma_config(config, create_env_fn)

    # add the callbacks
    # config["callbacks"] = {"on_train_result": on_train_result,
    #                        "on_episode_end": on_episode_end}
    config["callbacks"] = {"on_episode_end": on_episode_end}

    # config["eager_tracing"] = True
    # config["eager"] = True
    # config["eager_tracing"] = True

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    if args.kl_reward or args.l2_reward:
        runner = CustomPPOTrainer
    else:
        runner = args.algorithm

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': runner,
        'trial_name_creator': trial_str_creator,
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }
    return exp_dict, args


# def on_train_result(info):
#     """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
#     result = info["result"]
#
#     if 'policy_reward_mean' in result.keys():
#         if 'agent' in result['policy_reward_mean'].keys():
#             pendulum_reward = result['policy_reward_mean']['agent']


def on_episode_end(info):
    """Select the currently active adversary"""

    # store info about how many adversaries there are
    for env in info["env"].envs:
        env.select_new_adversary()
        episode = info["episode"]
        episode.custom_metrics["num_active_advs"] = env.adversary_range


if __name__ == "__main__":

    exp_dict, args = setup_exps(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/adv_robust/' \
                + date + '/' + args.exp_title
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()

    run_tune(**exp_dict, queue_trials=False, raise_on_failed_trial=False)

    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/goal_env'), date),
                                   args.exp_title)
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

                ray.shutdown()
                ray.init()

                visualize_adversaries(config, checkpoint_path, 100, output_path)

                if args.use_s3:
                    for i in range(4):
                        try:
                            p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                             "s3://sim2real/transfer_results/goal_env/{}/{}/{}".format(
                                                                                 date,
                                                                                 args.exp_title,
                                                                                 tune_name)).split(
                                ' '))
                            p1.wait(50)
                        except Exception as e:
                            print('This is the error ', e)
