"""Generate a cluster plot of where the different adversaries start the agent"""

"""Sample a bunch of observations from the observation space, plot the adversary action histogram for those runs"""

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

from visualize.mujoco.run_rollout import instantiate_rollout, DefaultMapping
from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from matplotlib.patches import Ellipse

from scipy.stats import norm


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
    num_arms = env.num_arms
    while sample_idx < num_samples:
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        obs = env.reset()['agent']
        if rllib_config['env'] == 'BernoulliMultiarmBandit':
            adv_obs = np.array([0.0])
        else:
            adv_obs = obs
        # we have an is_active key here
        if env.l2_reward and not env.l2_memory:
            multi_obs = {'agent': obs}
            adv_dict = {'adversary{}'.format(i): {'obs': adv_obs, 'is_active': np.array([1])} for i in range(num_adversaries)}
            multi_obs.update(adv_dict)
        else:
            multi_obs = {'agent': obs}
            adv_dict = {'adversary{}'.format(i): adv_obs for i in range(num_adversaries)}
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

    for adv_idx, (adversary, adv_dict) in enumerate(adversary_grid_dict.items()):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_ylabel("Arms", fontsize=10)
        ax.set_xlabel("Sampled Step", fontsize=10)
        sampled_actions = adv_dict['sampled_actions']
        heatmap = np.zeros((num_arms, len(sampled_actions)))
        for i, action in enumerate(sampled_actions):
            # action = np.sort(action)
            heatmap[:, i] = action

        plt.title('Arm Distributions for Adversary {}'.format(adv_idx), fontsize=12)
        output_str = '{}/{}_{}'.format(outdir, adversary, 'arm_distribution.png')
        plt.imshow(heatmap, aspect='auto')
        # plt.colorbar()
        plt.tight_layout()
        # plt.show()
        plt.savefig(output_str)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--num_samples', type=int, default=100, help='How many observations to sample')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/transfer_results/goal_env'),
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