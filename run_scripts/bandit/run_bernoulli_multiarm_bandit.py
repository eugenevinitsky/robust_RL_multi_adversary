from copy import deepcopy
import errno
from datetime import datetime
import os
import subprocess
import sys

import numpy as np
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
from envs.bernoulli_bandit import BernoulliMultiarmBandit
from visualize.goal_env.visualize_adversaries import visualize_adversaries
from visualize.pendulum.transfer_tests import run_transfer_tests, make_bernoulli_bandit_transfer_list
# from visualize.pendulum.visualize_adversaries import visualize_adversaries
from utils.pendulum_env_creator import make_create_env
from utils.parsers import init_parser, ray_parser, ma_env_parser
from utils.rllib_utils import get_config_from_path

from models.rocky_lstm import RockyLSTM

def setup_ma_config(config, create_env):
    env = create_env(config['env_config'])
    policies_to_train = ['agent']

    num_adversaries = config['env_config']['num_adv_strengths'] * config['env_config']['advs_per_strength']
    if num_adversaries == 0:
        return
    adv_policies = ['adversary' + str(i) for i in range(num_adversaries)]
    adversary_config = {"model": {'fcnet_hiddens': [64, 64], 'use_lstm': False}}
    agent_config = {"model": config['model']}
    if config['env_config']['kl_reward']:
        ModelCatalog.register_custom_action_dist("logits_dist", LogitsDist)
        adversary_config['model']['custom_action_dist'] = "logits_dist"
    # for both of these we need a graph that zeros out agents that weren't active
    if config['env_config']['kl_reward'] or (config['env_config']['l2_reward'] and not config['env_config']['l2_memory']):
        policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, agent_config)}
        policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in
                              range(num_adversaries)})
    else:
        policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, agent_config)}
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


def setup_exps(args, parser=None):
    if not parser:
        parser = init_parser()
    parser = ray_parser(parser)
    parser = ma_env_parser(parser)
    parser.add_argument('--env_name', default='pendulum', const='pendulum', nargs='?',
                        choices=['pendulum', 'hopper', 'cheetah'])
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--num_arms', type=int, default=3)
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
    parser.add_argument('--l2_memory', action='store_true', default=False,
                        help='If true we keep running mean statistics of the l2 score for each agent, allowing'
                             'us to not generate actions for each agent. This is noisier and more incorrect,'
                             'but WAY fast')
    parser.add_argument('--l2_memory_target_coeff', type=float, default=0.05,
                        help='The coefficient used to update the running mean if l2_memory is true')
    # Unused right now
    parser.add_argument('--kl_reward', action='store_true', default=False,
                        help='If true, each adversary gets a reward for being close to the adversaries in '
                             'KL space.')
    parser.add_argument('--kl_reward_coeff', type=float, default=1.0,
                        help='Scaling on the kl_reward')
    parser.add_argument('--num_concat_states', type=int, default=1,
                        help='This number sets how many previous states we concatenate into the observations')
    parser.add_argument('--concat_actions', action='store_true', default=False,
                        help='If true we concatenate prior actions into the state. This helps a lot for prediction.')
    parser.add_argument('--regret_formulation', action='store_true', default=False,
                        help='If true we compute reward as difference between obtained reward and expected reward'
                             'if the best arm was picked.')

    parser.add_argument('--lr', type=float, default=5e-3, help='The lower range that adversaries try'
                                                                      'to push you to')
    parser.add_argument('--gae_lambda', type=float, default=0.5, help='The upper range that adversaries try'
                                                                          'to push you to')
    args = parser.parse_args(args)

    if args.reward_range and args.num_adv_strengths * args.advs_per_strength <= 0:
        sys.exit('must specify number of strength levels, number of adversaries when using reward range')

    alg_run = args.algorithm

    if args.algorithm == 'PPO':
        config = deepcopy(DEFAULT_PPO_CONFIG)
        config['train_batch_size'] = 250000
        config['gamma'] = 0.99
        config['lambda'] = 0.3
        config['sgd_minibatch_size'] = 8192
        config['num_sgd_iter'] = 10
        config['observation_filter'] = 'NoFilter'


    if config['observation_filter'] == 'MeanStdFilter' and args.l2_reward:
        sys.exit('Mean std filter MUST be off if using the l2 reward')

    # Universal hyperparams
    config['num_workers'] = args.num_cpus
    config['seed'] = 0

    config['env_config']['horizon'] = args.horizon
    config['env_config']['num_arms'] = args.num_arms
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
    config['env_config']['l2_memory'] = args.l2_memory
    config['env_config']['l2_memory_target_coeff'] = args.l2_memory_target_coeff
    config['env_config']['concat_actions'] = args.concat_actions
    config['env_config']['num_concat_states'] = args.num_concat_states
    config['env_config']['regret_formulation'] = args.regret_formulation


    config['env_config']['run'] = alg_run

    ModelCatalog.register_custom_model("lstm", RockyLSTM)
    # TODO(@evinitsky) turn this on
    if args.use_lstm:
        config['model']['custom_model'] = 'lstm'
        config['model']['lstm_cell_size'] = 256

    env_name = "BernoulliMultiarmBandit"
    create_env_fn = make_create_env(BernoulliMultiarmBandit)

    config['env'] = env_name
    register_env(env_name, create_env_fn)

    setup_ma_config(config, create_env_fn)

    # add the callbacks
    config["callbacks"] = {"on_train_result": on_train_result,
                           "on_episode_end": on_episode_end}

    # config["eager_tracing"] = True
    # config["eager"] = True
    # config["eager_tracing"] = True

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    if (args.kl_reward) or (args.l2_reward and not args.l2_memory):
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


def on_train_result(info):
    """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
    result = info["result"]

    if info["result"]["config"]["env_config"]["l2_memory"]:
        trainer = info["trainer"]
        outputs = trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.get_observed_samples()))
        result_vec = np.zeros(outputs[0][0][0].shape)
        counts_vec = np.zeros(outputs[0][0][1].shape)
        # compute the mean weighted by how much each action was seen. We don't need to reweight the mean_vec,
        # it's already weighted in the env
        for output in outputs:
            mean_vec, counts = output[0]
            result_vec += mean_vec
            counts_vec += counts
        mean_result_vec = np.zeros(result_vec.shape)
        for i, row in enumerate(result_vec):
            if counts_vec[i] > 0:
                mean_result_vec[i] = row / counts_vec[i]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.update_global_action_mean(mean_result_vec)))

def on_episode_end(info):
    """Select the currently active adversary"""

    # store info about how many adversaries there are
    for env in info["env"].envs:
        env.select_new_adversary()
        # episode = info["episode"]
        # episode.custom_metrics["num_active_advs"] = env.adversary_range
        # episode.custom_metrics["action_space"] = env.adv_action_space.low.shape[0]
        # if env.l2_memory:
        #     # this sucks but custom metrics only supports single values
        #     for adv_index in range(env.local_l2_memory_array.shape[0]):
        #         episode.custom_metrics["num_observed_episodes_{}".format(adv_index)] = env.local_num_observed_l2_samples[adv_index]
        #         for action_index in range(env.local_l2_memory_array.shape[1]):
        #             episode.custom_metrics["adv_actions_{}_{}".format(adv_index, action_index)] = \
        #                 env.local_l2_memory_array[adv_index, action_index] * env.local_num_observed_l2_samples[adv_index]


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

    run_tune(**exp_dict, queue_trials=True, raise_on_failed_trial=False)
    num_adversaries = args.num_adv_strengths * args.advs_per_strength

    
    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/bandit'), date),
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
                run_transfer_tests(config, checkpoint_path, 1000, args.exp_title, output_path, run_list=make_bernoulli_bandit_transfer_list(args.num_arms))
                # visualize_adversaries(config, checkpoint_path, 100, output_path)

                if args.use_s3:
                    for i in range(4):
                        try:
                            p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                             "s3://sim2real/transfer_results/bandit/{}/{}/{}".format(
                                                                                 date,
                                                                                 args.exp_title,
                                                                                 tune_name)).split(
                                ' '))
                            p1.wait(50)
                        except Exception as e:
                            print('This is the error ', e)
