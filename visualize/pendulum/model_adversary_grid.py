import argparse
import errno
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns; sns.set()

from visualize.pendulum.run_rollout import instantiate_rollout, run_rollout
from utils.parsers import replay_parser
from utils.rllib_utils import get_config

"""Run through all of the model based adversaries and compute the score per adversary"""

def dict_func(env, options_dict):
    # figure out how many adversaries you have and initialize their grids
    results_dict = {}
    num_test_adversaries = options_dict['num_test_adversaries']
    env.num_adversaries = num_test_adversaries
    # we track our score for each adversary, the prediction error, and whether we guess that adversary correct
    results_dict['adversary_rew'] = np.zeros(num_test_adversaries)
    results_dict['prediction_error'] = np.zeros(num_test_adversaries)
    results_dict['adversary_guess'] = np.zeros(num_test_adversaries)

def pre_step_func(env, obs_dict):
    if isinstance(env.adv_observation_space, dict):
        multi_obs = {'adversary{}'.format(i): {'obs': obs_dict['pendulum'],
                                               'is_active': np.array([1])} for i in range(env.num_adversaries)}
    else:
        multi_obs = {'adversary{}'.format(i): obs_dict['pendulum'] for i in range(env.num_adversaries)}
    multi_obs.update({'pendulum': obs_dict['pendulum']})


def step_func(obs_dict, action_dict, logits_dict, results_dict, env):

    kl_grid = results_dict['kl_grid']

    for agent_id in obs_dict.keys():
        if agent_id != 'pendulum':
            # Now store the agent action in the corresponding grid
            action_bins = results_dict[agent_id]['action_bins']
            obs_bins = results_dict[agent_id]['obs_bins']

            heat_map = results_dict[agent_id]['grid']
            for action_loop_index, action in enumerate(action_dict[agent_id]):
                results_dict[agent_id]['action_list'].append(action[0])
                action_index = np.digitize(action, action_bins[action_loop_index, :]) - 1
                # digitize will set the right edge of the box to the wrong value
                if action_index == heat_map.shape[0]:
                    action_index -= 1
                for obs_loop_index, obs_elem in enumerate(obs_dict['pendulum'] * env.obs_norm):
                    obs_index = np.digitize(obs_elem, obs_bins[obs_loop_index, :]) - 1
                    if obs_index == heat_map.shape[1]:
                        obs_index -= 1

                    heat_map[action_index, obs_index, obs_loop_index] += 1

            # Now iterate through the agents and compute the kl_diff
            curr_id = int(agent_id.split('adversary')[1])
            your_logits = logits_dict[agent_id]
            mean, log_std = np.split(your_logits.numpy()[0], 2)
            for i in range(env.num_adversaries):
                # KL diff of something with itself is zero
                if i == curr_id:
                    pass
                # otherwise just compute the kl difference between the agents
                else:
                    other_logits = logits_dict['adversary{}'.format(i)]
                    other_mean, other_log_std = np.split(other_logits.numpy()[0], 2)
                    kl_diff = compute_kl_diff(mean, log_std, other_mean, other_log_std)
                    kl_grid[curr_id, i] += kl_diff


def on_result(results_dict, outdir):
    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # Plot the heatmap of the actions
    for adversary, adv_dict in results_dict.items():
        heat_map = adv_dict['grid']
        action_bins = adv_dict['action_bins']
        obs_bins = adv_dict['obs_bins']
        action_list = adv_dict['action_list']

        plt.figure()
        sns.distplot(action_list)
        output_str = '{}/{}'.format(outdir, adversary + 'action_histogram.png')
        plt.savefig(output_str)

        # x_label, y_label = env.transform_adversary_actions(bins)
        # ax = sns.heatmap(heat_map, annot=True, fmt="d")
        titles = ['x', 'y', 'thetadot']
        for i in range(heat_map.shape[-1]):
            plt.figure()
            # increasing the row index implies moving down on the y axis
            sns.heatmap(heat_map[:, :, i], yticklabels=np.round(action_bins[0], 1),
                        xticklabels=np.round(obs_bins[i], 1))
            plt.ylabel('Adversary actions')
            plt.xlabel(titles[i])
            output_str = '{}/{}'.format(outdir, adversary + 'action_heatmap_{}.png'.format(i))
            plt.savefig(output_str)

    # Plot the kl difference between agents
    plt.figure()
    sns.heatmap(results_dict['kl_grid'] / results_dict['total_steps'])
    output_str = '{}/{}'.format(outdir, 'kl_heatmap.png')
    plt.savefig(output_str)


def visualize_adversaries(rllib_config, checkpoint, num_test_adversaries, num_rollouts, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    options_dict = {'num_test_adversaries': num_test_adversaries}

    results_dict = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                               state_init, action_init, num_rollouts,
                               dict_func, pre_step_func, step_func, options_dict)
    on_result(results_dict, outdir)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--num_test_adversaries', type=int, default=20,
                        help='How many adversaries to actually test with')
    parser.add_argument('--output_dir', type=str, default='transfer_results',
                        help='Directory to output the files into')
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    rllib_config, checkpoint = get_config(args)
    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})

    ray.init(num_cpus=args.num_cpus)

    visualize_adversaries(rllib_config, checkpoint, args.num_test_adversaries, args.num_rollouts, args.output_dir)


if __name__ == '__main__':
    main()