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


def compute_kl_diff(mean, log_std, other_mean, other_log_std):
    """Compute the kl diff between agent and other agent"""
    std = np.exp(log_std)
    other_std = np.exp(other_log_std)
    return np.mean(other_log_std - log_std + (np.square(std) + np.square(mean - other_mean)) / (2.0 * np.square(other_std)) - 0.5)


def dict_func(env, options_dict):
    # figure out how many adversaries you have and initialize their grids
    num_adversaries = env.num_adversaries
    adversary_grid_dict = {}
    grid_size = options_dict['grid_size']
    kl_grid = np.zeros((num_adversaries, num_adversaries))
    for i in range(num_adversaries):
        adversary_str = 'adversary' + str(i)
        # each adversary grid is a map of agent action versus observation dimension
        adversary_grid = np.zeros((grid_size - 1, grid_size - 1, env.observation_space.low.shape[0])).astype(int)
        strength_grid = np.linspace(env.adv_action_space.low, env.adv_action_space.high, grid_size).T
        obs_grid = np.linspace(env.observation_space.low, env.observation_space.high, grid_size).T
        adversary_grid_dict[adversary_str] = {'grid': adversary_grid, 'action_bins': strength_grid,
                                              'obs_bins': obs_grid,
                                              'action_list': [],
                                              }
        adversary_grid_dict['kl_grid'] = kl_grid

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


def visualize_adversaries(rllib_config, checkpoint, grid_size, num_rollouts, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    options_dict = {'grid_size': grid_size}

    results_dict = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                               state_init, action_init, num_rollouts,
                               dict_func, pre_step_func, step_func, options_dict)
    on_result(results_dict, outdir)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--grid_size', type=int, default=50, help='How fine to make the adversary action grid')
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

    visualize_adversaries(rllib_config, checkpoint, args.grid_size, args.num_rollouts, args.output_dir)


if __name__ == '__main__':
    main()