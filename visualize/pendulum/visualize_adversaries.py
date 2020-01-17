import argparse
import errno
import logging
import os

from gym.spaces import Dict
import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns; sns.set()
from scipy.stats import chisquare

from ray.rllib.evaluation.episode import _flatten_action
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
        adversary_grid = np.zeros((grid_size - 1, grid_size - 1, env.high.shape[0])).astype(int)
        strength_grid = np.linspace(env.adv_action_space.low, env.adv_action_space.high, grid_size).T
        obs_grid = np.linspace(env.low, env.high, grid_size).T
        adversary_grid_dict[adversary_str] = {'grid': adversary_grid, 'action_bins': strength_grid,
                                              'obs_bins': obs_grid,
                                              'action_list': [],
                                              }
        adversary_grid_dict['kl_grid'] = kl_grid
    return adversary_grid_dict


def pre_step_func(env, obs_dict):
    if isinstance(env.adv_observation_space, dict):
        multi_obs = {'adversary{}'.format(i): {'obs': obs_dict['pendulum'],
                                               'is_active': np.array([1])} for i in range(env.num_adversaries)}
    else:
        multi_obs = {'adversary{}'.format(i): obs_dict['pendulum'] for i in range(env.num_adversaries)}
    multi_obs.update({'pendulum': obs_dict['pendulum']})
    return multi_obs


def step_func(obs_dict, action_dict, logits_dict, results_dict, env):

    kl_grid = results_dict['kl_grid']

    for agent_id in obs_dict.keys():
        if agent_id != 'pendulum':
            # Now store the agent action in the corresponding grid
            action_bins = results_dict[agent_id]['action_bins']
            obs_bins = results_dict[agent_id]['obs_bins']

            heat_map = results_dict[agent_id]['grid']
            for action_loop_index, action in enumerate(action_dict[agent_id]):
                results_dict[agent_id]['action_list'].append(action)
                action_index = np.digitize(action, action_bins[action_loop_index, :]) - 1
                # digitize will set the right edge of the box to the wrong value
                if action_index == heat_map.shape[0]:
                    action_index -= 1
                for obs_loop_index, obs_elem in enumerate(env._get_obs()):
                    obs_index = np.digitize(obs_elem, obs_bins[obs_loop_index, :]) - 1
                    if obs_index == heat_map.shape[1]:
                        obs_index -= 1

                    heat_map[action_index, obs_index, obs_loop_index] += 1

            # Now iterate through the agents and compute the kl_diff
            curr_id = int(agent_id.split('adversary')[1])
            if len(logits_dict) > 0:
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
    adversary_dict = {key: val for key, val in results_dict.items() if 'adversary' in key}
    for adversary, adv_dict in adversary_dict.items():
        heat_map = adv_dict['grid']
        action_bins = adv_dict['action_bins']
        obs_bins = adv_dict['obs_bins']
        action_list = adv_dict['action_list']

        fig = plt.figure()
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
        plt.close(fig)

    # Plot the kl difference between agents
    fig = plt.figure()
    sns.heatmap(results_dict['kl_grid'] / results_dict['total_steps'])
    output_str = '{}/{}'.format(outdir, 'kl_heatmap.png')
    plt.savefig(output_str)
    plt.close(fig)


def visualize_adversaries(rllib_config, checkpoint, grid_size, num_rollouts, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    options_dict = {'grid_size': grid_size}
    results_dict = dict_func(env, options_dict=options_dict)

    results_dict = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                               state_init, action_init, num_rollouts,
                               pre_step_func, step_func, None, results_dict)
    on_result(results_dict, outdir)

    # TODO(@evinitsky) clean this up doood. Its not in the same abstraction as the other code
    # generate heatmap of adversary actions for all possible states
    adversary_grid_dict = {}
    mapping_cache = {}  # in case policy_agent_mapping is stochastic
    if 'num_adversaries' in rllib_config['env_config'].keys() and rllib_config['env_config']['num_concat_states'] == 1:
        num_adversaries = rllib_config['env_config']['num_adversaries'] * rllib_config['env_config'].get('num_adv_per_strength', 1)
        for i in range(num_adversaries):
            adversary_str = 'adversary' + str(i)
            # each adversary grid is a map of agent action versus observation dimension
            adversary_grid = np.zeros((grid_size, grid_size, env._get_obs().shape[0])).astype(int)
            strength_grid = np.linspace(env.adv_action_space.low, env.adv_action_space.high, grid_size).T
            obs_grid = np.linspace(env.observation_space.low, env.observation_space.high, grid_size).T
            adversary_grid_dict[adversary_str] = {'grid': adversary_grid, 'action_bins': strength_grid,
                                                  'obs_bins': obs_grid,
                                                  'action_list': []}

        for r_iter in range(num_rollouts):
            obs = env.reset()
            if isinstance(env.adv_observation_space, Dict):
                multi_obs = {'adversary{}'.format(i): {'obs': obs['pendulum'], 'is_active': np.array([1])} for i in
                             range(env.num_adversaries)}
            else:
                multi_obs = {'adversary{}'.format(i): obs['pendulum'] for i in range(env.num_adversaries)}
            for agent_id, a_obs in multi_obs.items():
                if agent_id == 'pendulum':
                    continue
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))

                all_obs_comb = np.asarray(np.meshgrid(*np.split(obs_grid, obs_grid.shape[0]))).T.reshape(-1,
                                                                                                         obs_grid.shape[0])
                for obs in all_obs_comb:
                    if isinstance(env.adv_observation_space, Dict):
                        multi_obs = {'obs': obs, 'is_active': np.array([1])}
                    else:
                        multi_obs = obs
                    flat_action = _flatten_action(multi_obs)
                    try:
                        a_action = agent.compute_action(flat_action, prev_action=[0], prev_reward=0, policy_id=policy_id)
                    except Exception as e:
                        print(e)
                        a_action = agent.compute_action(flat_action, prev_action=[0], prev_reward=0, policy_id=policy_id)

                    # handle the tuple case
                    if len(a_action) > 1:
                        if isinstance(a_action[0], np.ndarray):
                            a_action[0] = a_action[0].flatten()

                    # Now store the agent action in the corresponding grid
                    if agent_id != 'pendulum':
                        action_bins = adversary_grid_dict[agent_id]['action_bins']
                        obs_bins = adversary_grid_dict[agent_id]['obs_bins']

                        heat_map = adversary_grid_dict[agent_id]['grid']
                        for action_loop_index, action in enumerate(a_action):
                            adversary_grid_dict[agent_id]['action_list'].append(a_action[0])
                            action_index = np.digitize(action, action_bins[action_loop_index, :]) - 1
                            # digitize will set the right edge of the box to the wrong value
                            if action_index == heat_map.shape[0]:
                                action_index -= 1
                            # remove the action since we don't want to plot it
                            if rllib_config['env_config']['concat_actions']:
                                obs = obs[:-1]
                            for obs_loop_index, obs_elem in enumerate(obs):
                                obs_index = np.digitize(obs_elem, obs_bins[obs_loop_index, :]) - 1
                                if obs_index == heat_map.shape[1]:
                                    obs_index -= 1

                                heat_map[action_index, obs_index, obs_loop_index] += 1

        print(heat_map[:, :, 0])
        for adversary, adv_dict in adversary_grid_dict.items():
            heat_map = adv_dict['grid']
            action_bins = adv_dict['action_bins']
            obs_bins = adv_dict['obs_bins']
            action_list = adv_dict['action_list']

            fig = plt.figure()
            sns.distplot(action_list)
            output_str = '{}/{}'.format(outdir, adversary + 'action_histogram_all.png')
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
                output_str = '{}/{}'.format(outdir, adversary + 'action_heatmap_{}_all.png'.format(i))
                plt.savefig(output_str)
            plt.close(fig)

        # Compute state dependence, i.e. whether p(action|state) != p(action)
        for adversary, adv_dict in adversary_grid_dict.items():
            heat_map = adv_dict['grid']

            p_vals = np.zeros((heat_map.shape[1], heat_map.shape[-1]))  # shape state value x state type
            for i in range(heat_map.shape[-1]):
                prob_action = np.sum(heat_map[:, :, i], axis=1)  # sum across all states
                prob_action = prob_action / heat_map.shape[1]  # normalize probability
                prob_action = np.squeeze(prob_action)

                for j in range(heat_map.shape[1]):
                    # remove bins with 0 count for more accurate chi-squared value
                    prob_action_filtered = prob_action[~(heat_map[:, j, i] == 0)]
                    prob_actionstate_filtered = heat_map[~(heat_map[:, j, i] == 0), j, i]
                    chisq, p_val = chisquare(prob_action_filtered, f_exp=prob_actionstate_filtered)
                    p_vals[j, i] = p_val

                fig = plt.figure()
                plt.plot(obs_bins[i], p_vals[:, i])
                plt.ylabel('P-val')
                plt.xlabel(titles[i])
                output_str = '{}/{}'.format(outdir, adversary + 'p_vals_heatmap_{}.png'.format(i))
                plt.savefig(output_str)
                plt.close(fig)
            print(p_vals)


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