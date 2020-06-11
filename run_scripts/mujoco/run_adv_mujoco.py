from copy import deepcopy
import errno
from datetime import datetime
import os
import subprocess
import sys

import numpy as np
import psutil
import pytz
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as DEFAULT_PPO_CONFIG
from ray.rllib.agents.sac.sac import DEFAULT_CONFIG as DEFAULT_SAC_CONFIG

from ray.rllib.agents.ddpg.td3 import TD3_DEFAULT_CONFIG as DEFAULT_TD3_CONFIG
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy

from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import Trainable
from ray.tune.logger import pretty_print
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from algorithms.multi_active_ppo import CustomPPOPolicy, CustomPPOTrainer
from algorithms.custom_kl_distribution import LogitsDist
from envs.mujoco.adv_hopper import AdvMAHopper
from envs.mujoco.adv_inverted_pendulum_env import AdvMAPendulumEnv
from envs.mujoco.adv_cheetah import AdvMAHalfCheetahEnv
from envs.mujoco.adv_ant import AdvMAAnt

from visualize.mujoco.transfer_tests import run_transfer_tests
from visualize.mujoco.action_sampler import sample_actions
# from visualize.mujoco.visualize_adversaries import visualize_adversaries
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
    adversary_config = {"model": {'fcnet_hiddens': [64, 64], 'use_lstm': False}, "entropy_coeff": config['env_config']['entropy_coeff']}
    if config['env_config']['run'] == 'PPO':
        if config['env_config']['kl_reward']:
            ModelCatalog.register_custom_action_dist("logits_dist", LogitsDist)
            adversary_config['model']['custom_action_dist'] = "logits_dist"
        # for both of these we need a graph that zeros out agents that weren't active
        if config['env_config']['kl_reward'] or (config['env_config']['l2_reward'] and not config['env_config']['l2_memory']):
            policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, {})}
            policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
                                                    env.adv_action_space, adversary_config) for i in
                                  range(num_adversaries)})
        else:
            policy_graphs = {'agent': (PPOTFPolicy, env.observation_space, env.action_space, {})}
            policy_graphs.update({adv_policies[i]: (PPOTFPolicy, env.adv_observation_space,
                                                    env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    elif config['env_config']['run'] == 'TD3':
        policy_graphs = {'agent': (DDPGTFPolicy, env.observation_space, env.action_space, {})}
        policy_graphs.update({adv_policies[i]: (DDPGTFPolicy, env.adv_observation_space,
                                                env.adv_action_space, adversary_config) for i in range(num_adversaries)})
    
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
    parser.add_argument('--env_name', default='pendulum', const='pendulum', nargs='?', choices=['pendulum', 'hopper', 'cheetah', 'ant'])
    parser.add_argument('--algorithm', default='PPO', type=str, help='Options are PPO, SAC, TD3')
    parser.add_argument('--custom_ppo', action='store_true', default=False, help='If true, we use the PPO with a KL penalty')
    parser.add_argument('--num_adv_strengths', type=int, default=1, help='Number of adversary strength ranges. '
                                                                         'Multiply this by `advs_per_strength` to get the total number of adversaries'
                                                                         'Default single agent trained with RARL')
    parser.add_argument('--advs_per_strength', type=int, default=1, help='How many adversaries exist at each strength level')
    parser.add_argument('--adv_strength', type=float, default=5.0, help='Strength of active adversaries in the env')
    parser.add_argument('--alternate_training', action='store_true', default=False)
    parser.add_argument('--curriculum', action='store_true', default=False,
                        help='If true, the number of adversaries is increased every `adv_incr_freq` steps that'
                             'we are above goal score')
    parser.add_argument('--goal_score', type=float, default=3000.0,
                        help='This is the score we need to maintain for `adv_incr_freq steps before the number'
                             'of adversaries increase')
    parser.add_argument('--adv_incr_freq', type=int, default=20,
                        help='If you stay above `goal_score` for this many steps, the number of adversaries'
                             'will increase')
    parser.add_argument('--num_concat_states', type=int, default=1,
                        help='This number sets how many previous states we concatenate into the observations')
    parser.add_argument('--concat_actions', action='store_true', default=False,
                        help='If true we concatenate prior actions into the state. This helps a lot for prediction.')
    parser.add_argument('--domain_randomization', action='store_true', default=False,
                        help='If true we use vanilla domain randomization over the transfer task.')
    parser.add_argument('--extreme_domain_randomization', action='store_true', default=False,
                        help='If true we use domain randomization across different joints/links as well')
    parser.add_argument('--cheating', action='store_true', default=False,
                        help='Enabled with domain randomization, will provide the learner with the transfer params.')
    parser.add_argument('--reward_range', action='store_true', default=False,
                        help='If true, the adversaries try to get agents to goals evenly spaced between `low_reward`'
                             'and `high_reward')
    parser.add_argument('--num_adv_rews', type=int, default=1, help='Number of adversary rews ranges if reward ranges is on')
    parser.add_argument('--advs_per_rew', type=int, default=1,
                        help='How many adversaries exist at a given reward level')
    parser.add_argument('--low_reward', type=float, default=0.0, help='The lower range that adversaries try'
                                                                      'to push you to')
    parser.add_argument('--high_reward', type=float, default=4000.0, help='The upper range that adversaries try'
                                                                          'to push you to')
    parser.add_argument('--l2_reward', action='store_true', default=False,
                        help='If true, each adversary gets a reward for being close to the adversaries. This '
                             'is NOT a supervised loss')
    parser.add_argument('--l2_reward_coeff', type=float, default=0.5,
                        help='Scaling on the l2_reward')
    parser.add_argument('--l2_in_tranche', action='store_true', default=False,
                        help='If this is true, you only compare l2 values for adversaries that have the same reward '
                             'goal as you ')
    parser.add_argument('--l2_memory', action='store_true', default=False,
                        help='If true we keep running mean statistics of the l2 score for each agent, allowing'
                             'us to not generate actions for each agent. This is noisier and more incorrect,'
                             'but WAY faster')
    parser.add_argument('--l2_memory_target_coeff', type=float, default=0.05,
                        help='The coefficient used to update the running mean if l2_memory is true. '
                             '1 / this value sets an approximate time scale for updating. Keep it nice and low.')

    parser.add_argument('--kl_reward', action='store_true', default=False,
                        help='If true, each adversary gets a reward for being close to the adversaries in '
                             'KL space.')
    parser.add_argument('--kl_reward_coeff',  type=float, default=1.0,
                        help='Scaling on the kl_reward')
    parser.add_argument('--no_end_if_fall', action='store_true', default=False,
                        help='If true, the env continues even after a fall ')
    parser.add_argument('--adv_all_actions', action='store_true', default=False,
                        help='If true we apply perturbations to the actions instead of the RARL parametrization')
    parser.add_argument('--entropy_coeff', type=float, default=0.0,
                        help='If you want to penalize entropy, set this to a negative value')
    parser.add_argument('--clip_actions', action='store_true', default=False,
                        help='If true, the sum of the adversary and agent actions is clipped')

    parser.add_argument('--lambda_val', type=float, default=0.9,
                        help='PPO lambda value')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='PPO lambda value')

    args = parser.parse_args(args)

    if args.alternate_training and args.advs_per_strength > 1:
        sys.exit('You can only have 1 adversary if you are alternating training')
    if args.cheating and not args.domain_randomization:
        sys.exit('cheating should not be enabled without domain randomization' )
    if args.reward_range and args.num_adv_strengths * args.advs_per_strength <= 0:
        sys.exit('must specify number of strength levels, number of adversaries when using reward range')
    if (args.num_adv_strengths * args.advs_per_strength != args.num_adv_rews * args.advs_per_rew) and args.reward_range:
        sys.exit('Your number of adversaries per reward range must match the total number of adversaries')
    if args.grid_search and args.seed_search:
        sys.exit('You can\'t both sweed seeds and grid search')

    alg_run = args.algorithm

    if args.algorithm == 'PPO':
        config = deepcopy(DEFAULT_PPO_CONFIG)
        config['seed'] = 0
        config['train_batch_size'] = args.train_batch_size
        config['gamma'] = 0.995
        config['observation_filter'] = 'MeanStdFilter'
        if args.env_name == 'cheetah':
            config['kl_coeff'] = 1.0
            config['vf_loss_coeff'] = 0.5
            config['clip_param'] = 0.2
            config['grad_clip'] = 0.5
            config['gamma'] = 0.99
        config['vf_clip_param'] = 100.0
        if args.grid_search:
            if args.env_name == 'cheetah':
                config['lambda'] = tune.grid_search([0.9, 0.95, 1.0])
                config ['lr'] = tune.grid_search([3e-4, 5e-4])
            else:
                config['lambda'] = tune.grid_search([0.5, 0.9, 1.0])
                config['lr'] = tune.grid_search([5e-5, 5e-4])

        elif args.seed_search:
            config['seed'] = tune.grid_search([i for i in range(10)])
            config['lr'] = args.lr
            config['lambda'] = args.lambda_val
        else:
            if args.env_name == 'hopper':
                config['lambda'] = 0.9
                config['lr'] = 5e-4
            else:
                config['lambda'] = 0.9
                config['lr'] = 5e-5
        config['sgd_minibatch_size'] = 64 * max(int(args.train_batch_size / 1e4), 1)
        if args.use_lstm:
            config['sgd_minibatch_size'] *= 5
        config['num_sgd_iter'] = 10
    elif args.algorithm == 'SAC':
        config = DEFAULT_SAC_CONFIG
        config['target_network_update_freq'] = 1
    elif args.algorithm == 'TD3':
        config = DEFAULT_TD3_CONFIG
        # === Exploration ===
        config['learning_starts'] = 10000
        config['pure_exploration_steps'] = 10000
        if args.grid_search:
            config["actor_lr"] = tune.grid_search([1e-3, 1e-4, 1e-5])
            config["critic_lr"] = tune.grid_search([1e-3, 1e-4, 1e-5])
            config["tau"] = tune.grid_search([5e-3, 5e-4])

        elif args.seed_search:
            config['seed'] = tune.grid_search([i for i in range(9)])
        # === Evaluation ===
        config['evaluation_interval'] = 5
        config['evaluation_num_episodes'] = 10
    else:
        sys.exit('Only PPO, TD3, and SAC are supported')

    if config['observation_filter'] == 'MeanStdFilter' and args.l2_reward:
        sys.exit('Mean std filter MUST be off if using the l2 reward')

    # Universal hyperparams
    config['num_workers'] = args.num_cpus
    config["batch_mode"] = "complete_episodes"

    # config['num_adversaries'] = args.num_adv
    # config['kl_diff_weight'] = args.kl_diff_weight
    # config['kl_diff_target'] = args.kl_diff_target
    # config['kl_diff_clip'] = 5.0

    config['env_config']['num_adv_strengths'] = args.num_adv_strengths
    config['env_config']['advs_per_strength'] = args.advs_per_strength
    config['env_config']['adversary_strength'] = args.adv_strength
    config['env_config']['reward_range'] = args.reward_range
    config['env_config']['num_adv_rews'] = args.num_adv_rews
    config['env_config']['advs_per_rew'] = args.advs_per_rew

    config['env_config']['low_reward'] = args.low_reward
    config['env_config']['high_reward'] = args.high_reward
    config['env_config']['curriculum'] = args.curriculum
    config['env_config']['goal_score'] = args.goal_score
    config['env_config']['adv_incr_freq'] = args.adv_incr_freq
    config['env_config']['concat_actions'] = args.concat_actions
    config['env_config']['num_concat_states'] = args.num_concat_states
    config['env_config']['domain_randomization'] = args.domain_randomization
    config['env_config']['extreme_domain_randomization'] = args.extreme_domain_randomization
    config['env_config']['cheating'] = args.cheating
    config['env_config']['l2_reward'] = args.l2_reward
    config['env_config']['kl_reward'] = args.kl_reward
    config['env_config']['l2_reward_coeff'] = args.l2_reward_coeff
    config['env_config']['kl_reward_coeff'] = args.kl_reward_coeff
    config['env_config']['l2_in_tranche'] = args.l2_in_tranche
    config['env_config']['l2_memory'] = args.l2_memory
    config['env_config']['l2_memory_target_coeff'] = args.l2_memory_target_coeff
    config['env_config']['no_end_if_fall'] = args.no_end_if_fall
    config['env_config']['adv_all_actions'] = args.adv_all_actions
    config['env_config']['entropy_coeff'] = args.entropy_coeff
    config['env_config']['clip_actions'] = args.clip_actions

    config['env_config']['run'] = alg_run

    ModelCatalog.register_custom_model("rnn", LSTM)
    config['model']['fcnet_hiddens'] = [64, 64]
    if args.use_lstm:
        config['model']['fcnet_hiddens'] = [64]
        config['model']['use_lstm'] = False
        config['model']['lstm_use_prev_action_reward'] = True
        config['model']['lstm_cell_size'] = 64

    if args.env_name == "pendulum":
        env_name = "MAPendulumEnv"
        env_tag = "pendulum"
        create_env_fn = make_create_env(AdvMAPendulumEnv)
    elif args.env_name == "hopper":
        env_name = "MAHopperEnv"
        env_tag = "hopper"
        create_env_fn = make_create_env(AdvMAHopper)
    elif args.env_name == "cheetah":
        env_name = "MACheetahEnv"
        env_tag = "cheetah"
        create_env_fn = make_create_env(AdvMAHalfCheetahEnv)
    elif args.env_name == "ant":
        env_name = "MAAntEnv"
        env_tag = "ant"
        create_env_fn = make_create_env(AdvMAAnt)

    config['env'] = env_name
    register_env(env_name, create_env_fn)

    setup_ma_config(config, create_env_fn)

    # add the callbacks
    config["callbacks"] = {"on_train_result": on_train_result,
                           "on_episode_end": on_episode_end}

    # create a custom string that makes looking at the experiment names easier
    def trial_str_creator(trial):
        return "{}_{}".format(trial.trainable_name, trial.experiment_tag)

    if args.kl_reward or (args.l2_reward and not args.l2_memory):
        runner = CustomPPOTrainer
    else:
        runner = args.algorithm

    stop_dict = {}
    if args.algorithm == 'PPO':
        stop_dict.update({
            'training_iteration': args.num_iters
        })
    elif args.algorithm == 'TD3':
        stop_dict.update({
            'timesteps_total': args.num_iters * 10000
        })

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': runner,
        'trial_name_creator': trial_str_creator,
        # 'checkpoint_freq': args.checkpoint_freq,
        'checkpoint_at_end': True,
        'stop': stop_dict,
        'config': config,
        'num_samples': args.num_samples,
    }
    return exp_dict, args


def on_train_result(info):
    """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
    result = info["result"]

    if 'policy_reward_mean' in result.keys() and result["config"]["env_config"]["curriculum"]:
        if 'agent' in result['policy_reward_mean'].keys():
            pendulum_reward = result['policy_reward_mean']['agent']
            trainer = info["trainer"]

            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.update_curriculum(pendulum_reward)))

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
    if hasattr(info["env"], 'envs'):
        env = info["env"].envs[0]
        env.select_new_adversary()
        if hasattr(env, 'domain_randomization') and env.domain_randomization:
            env.randomize_domain()
        elif hasattr(env, 'extreme_domain_randomization') and env.extreme_domain_randomization:
            env.extreme_randomize_domain()
        episode = info["episode"]
        episode.custom_metrics["num_active_advs"] = env.adversary_range


class AlternateTraining(Trainable):
    def _setup(self, config):
        self.config = config
        self.env = config['env']
        agent_config = self.config
        adv_config = deepcopy(self.config)
        agent_config['multiagent']['policies_to_train'] = ['agent']
        adv_config['multiagent']['policies_to_train'] = ['adversary0']

        self.agent_trainer = PPOTrainer(env=self.env, config=agent_config)
        self.adv_trainer = PPOTrainer(env=self.env, config=adv_config)

    def _train(self):
        # improve the Adversary policy
        print("-- Adversary Training --")
        print(pretty_print(self.adv_trainer.train()))

        # swap weights to synchronize
        self.agent_trainer.set_weights(self.adv_trainer.get_weights(["adversary0"]))

        # improve the Agent policy
        print("-- Agent Training --")
        output = self.agent_trainer.train()
        print(pretty_print(output))

        # swap weights to synchronize
        self.adv_trainer.set_weights(self.agent_trainer.get_weights(["agent"]))
        return output

    def _save(self, tmp_checkpoint_dir):
        return self.agent_trainer._save(tmp_checkpoint_dir)


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

    if args.alternate_training:
        exp_dict['run_or_experiment'] = AlternateTraining
    run_tune(**exp_dict, queue_trials=False, raise_on_failed_trial=False)

    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/adv_robust'), date), args.exp_title)
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
            # if "checkpoint_{}".format(args.num_iters) in dirpath:
            if "checkpoint" in dirpath and dirpath.split('/')[-3] == args.exp_title:

                # grab the experiment name
                folder = os.path.dirname(dirpath)
                tune_name = folder.split("/")[-1]
                outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.expanduser(os.path.join(outer_folder, "visualize/transfer_test.py"))
                config, checkpoint_path = get_config_from_path(folder, dirpath.split('_')[-1])

                test_list = []
                if config['env'] == "MAPendulumEnv":
                    from visualize.mujoco.transfer_tests import pendulum_run_list
                    run_list = pendulum_run_list
                elif config['env'] == "MAHopperEnv":
                    from visualize.mujoco.transfer_tests import hopper_run_list, hopper_test_list
                    run_list = hopper_run_list
                    test_list = hopper_test_list
                elif config['env'] == "MACheetahEnv":
                    from visualize.mujoco.transfer_tests import cheetah_run_list, cheetah_test_list
                    run_list = cheetah_run_list
                    test_list = cheetah_test_list
                elif config['env'] == "MAAntEnv":
                    from visualize.mujoco.transfer_tests import ant_run_list, ant_test_list
                    run_list = ant_run_list
                    test_list = ant_test_list

                ray.shutdown()
                ray.init()
                run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=run_list)
                if len(test_list) > 0:
                    run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=test_list, is_test=True)

                sample_actions(config, checkpoint_path, min(2 * args.train_batch_size, 20000), output_path)

                if args.use_s3:
                    # visualize_adversaries(config, checkpoint_path, 10, 100, output_path)
                    for i in range(4):
                        try:
                            p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                             "s3://sim2real/transfer_results/adv_robust/{}/{}/{}".format(date,
                                                                                                                              args.exp_title,
                                                                                                                              tune_name)).split(
                                ' '))
                            p1.wait(50)
                        except Exception as e:
                            print('This is the error ', e)
