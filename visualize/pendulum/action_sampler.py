"""Sample a bunch of observations from the observation space, plot the adversary action histogram for those runs"""

import argparse
import collections
import errno
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.models.tf.tf_action_dist import DiagGaussian

from visualize.pendulum.run_rollout import instantiate_rollout, DefaultMapping
from utils.parsers import replay_parser
from utils.rllib_utils import get_config


def compute_kl_diff(mean, log_std, other_mean, other_log_std):
    """Compute the kl diff between agent and other agent"""
    std = np.exp(log_std)
    other_std = np.exp(other_log_std)
    return np.mean(other_log_std - log_std + (np.square(std) + np.square(mean - other_mean)) / (2.0 * np.square(other_std)) - 0.5)


def sample_actions(rllib_config, checkpoint, num_samples, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    # figure out how many adversaries you have and initialize their grids
    num_adversaries = env.num_adv_strengths * env.advs_per_strength
    adversary_grid_dict = {}
    for i in range(num_adversaries):
        adversary_str = 'adversary' + str(i)
        # each adversary grid is a map of agent action versus observation dimension
        adversary_grid_dict[adversary_str] = {'sampled_actions': np.zeros((num_samples, env.adv_action_space.shape[0]))}

    mapping_cache = {}  # in case policy_agent_mapping is stochastic

    sample_idx = 0
    while sample_idx < num_samples:
        obs = env.reset()['agent']
        done = False
        while not done:
            multi_obs = {'adversary{}'.format(i): obs for i in range(num_adversaries)}
            multi_obs['agent'] = obs
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
                            policy_id=policy_id)
                        if agent_id != 'agent':
                            adversary_grid_dict[agent_id]['sampled_actions'][sample_idx] = a_action
                        action_dict[agent_id] = a_action
            new_dict = {}
            new_dict.update({'agent': action_dict['agent']})
            obs, reward, done, info = env.step(new_dict)
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
    for adversary, adv_dict in adversary_grid_dict.items():
        sampled_actions = adv_dict['sampled_actions']
        for action_idx in range(sampled_actions.shape[-1]):
            fig = plt.figure()
            plt.hist(sampled_actions[:, action_idx])
            output_str = '{}/{}'.format(outdir, adversary + 'action_{}_histogram.png'.format(action_idx))
            plt.xlabel('Action magnitude')
            plt.ylabel('Frequency')
            plt.title('Histograms of actions over {} sampled obs'.format(num_samples))
            plt.savefig(output_str)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--num_samples', type=int, default=1000, help='How many observations to sample')
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

    sample_actions(rllib_config, checkpoint, args.num_samples, args.output_dir)

if __name__ == '__main__':
    main()