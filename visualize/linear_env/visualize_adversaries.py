"""Sample a bunch of observations from the observation space, plot their eigenvalues"""

import argparse
import collections
import errno
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import ray
from ray.rllib.evaluation.episode import _flatten_action

from visualize.pendulum.run_rollout import instantiate_rollout, DefaultMapping
from utils.parsers import replay_parser
from utils.rllib_utils import get_config


def visualize_adversaries(rllib_config, checkpoint, num_samples, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    # figure out how many adversaries you have and initialize their grids
    num_adversaries = env.num_adv_strengths * env.advs_per_strength
    adversary_grid_dict = {}
    for i in range(num_adversaries):
        adversary_str = 'adversary' + str(i)
        # each adversary grid is a map of agent action versus observation dimension
        adversary_grid_dict[adversary_str] = {'sampled_actions': []}

    mapping_cache = {}  # in case policy_agent_mapping is stochastic

    sample_idx = 0
    while sample_idx < num_samples:
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        env.reset()
        multi_obs = {}
        adv_dict = {'adversary{}'.format(i): env.curr_pos for i in range(num_adversaries)}
        multi_obs.update(adv_dict)
        action_dict = {}
        for agent_id, a_obs in multi_obs.items():
            if a_obs is not None:
                policy_id = mapping_cache.setdefault(
                    agent_id, policy_agent_mapping(agent_id))
                p_use_lstm = use_lstm[policy_id]
                if not p_use_lstm:
                    flat_obs = _flatten_action(a_obs)
                    a_action = agent.compute_action(
                        flat_obs,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                    if agent_id != 'agent':
                        adversary_grid_dict[agent_id]['sampled_actions'].append(a_action)
                    action_dict[agent_id] = a_action
        sample_idx += 1


    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # Plot the histogram of the actions
    colors = cm.rainbow(np.linspace(0, 1, num_adversaries))

    for adversary, adv_dict in adversary_grid_dict.items():
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim([-1 * env.dim * env.adversary_strength, 1 * env.dim * env.adversary_strength])
        ax.set_ylim([-1 * env.dim * env.adversary_strength, 1 * env.dim * env.adversary_strength])
        i = 0
        handle_list = []
        real_vals = []
        img_vals = []
        for matrix in adv_dict['sampled_actions']:
            eigenvals = np.linalg.eigvals(matrix.reshape((env.dim, env.dim)))
            for eig in eigenvals:
                real_vals.append(np.real(eig))
                img_vals.append(np.imag(eig))
        handle_list.append(plt.scatter(real_vals, img_vals, color=colors[i], label=adversary, s=15))
        plt.title('Scatter of eigenvalues for {}'.format(adversary))
        i += 1
        output_str = '{}/{}'.format(outdir, '{}_action_histogram.png'.format(adversary))
        # legends = ['goal'] + ['adversary{}'.format(i) for i in range(len(adversary_grid_dict))]
        plt.legend(handles=handle_list)
        # plt.legend(legends)
        plt.savefig(output_str)
        plt.close(fig)

    # now create a bar chart of the eigenvalue magnitude
    fig = plt.figure(figsize=(8, 8))
    indices = np.arange(len(adversary_grid_dict))
    magnitudes = []
    for adversary, adv_dict in adversary_grid_dict.items():
        magnitude = []
        for matrix in adv_dict['sampled_actions']:
            eigenvals = np.linalg.eigvals(matrix.reshape((env.dim, env.dim)))
            for eig in eigenvals:
                magnitude.append(np.linalg.norm(eig))
        magnitudes.append([adversary, np.mean(magnitude)])
    output_str = '{}/{}'.format(outdir, 'action_bar_plot.png')
    plt.bar(indices, [val[1] for val in magnitudes])
    plt.xticks(indices, [val[0] for val in magnitudes])
    plt.savefig(output_str)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--num_samples', type=int, default=100, help='How many observations to sample')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/transfer_results/linear_env'),
                        help='Directory to output the files into')
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    rllib_config, checkpoint = get_config(args)
    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})

    ray.init(num_cpus=args.num_cpus)

    visualize_adversaries(rllib_config, checkpoint, args.num_samples, args.output_dir)

if __name__ == '__main__':
    main()