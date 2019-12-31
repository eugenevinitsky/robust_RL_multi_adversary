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
import seaborn as sns; sns.set()

from visualize.rollout import instantiate_rollout, DefaultMapping
from utils.parsers import replay_parser
from utils.rllib_utils import get_config


def compute_kl_diff(log_std, mean, other_log_std, other_mean):
    """Compute the kl diff between agent and other agent"""
    std = np.exp(log_std)
    other_std = np.exp(other_log_std)
    return np.mean(
        other_log_std - log_std +
        (np.square(std) + np.square(mean - other_mean)) /
        (2.0 * np.square(other_std)) - 0.5,
    )


def visualize_adversaries(rllib_config, checkpoint, grid_size, num_rollouts, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint, False)

    # figure out how many adversaries you have and initialize their grids
    num_adversaries = env.num_adversaries
    adversary_grid_dict = {}
    kl_grid = np.zeros((num_adversaries, num_adversaries))
    for i in range(num_adversaries):
        adversary_str = 'adversary' + str(i)
        adversary_grid = np.zeros((grid_size, grid_size)).astype(int)
        strength_list = np.linspace(env.adv_action_space.low, env.adv_action_space.high, grid_size).T
        adversary_grid_dict[adversary_str] = {'grid': adversary_grid, 'bins': strength_list}

    total_steps = 0

    # actually do the rollout
    for r_itr in range(num_rollouts):
        print('On iteration {}'.format(r_itr))
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        obs = env.reset()
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        step_num = 0
        while not done:
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            obs = multi_obs['robot']
            multi_obs = {'adversary{}'.format(i): obs for i in range(env.num_adversaries)}
            multi_obs.update({'robot': obs})
            action_dict = {}
            logits_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    policy = agent.get_policy(policy_id)
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        prev_action = _flatten_action(prev_actions[agent_id])
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_action,
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state

                        logits, _ = policy.model.from_batch({"obs": a_obs[np.newaxis, :],
                                                             "prev_action": prev_action})
                    else:
                        logits, _ = policy.model.from_batch({"obs": a_obs[np.newaxis, :]})
                        prev_action = _flatten_action(prev_actions[agent_id])
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_action,
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)

                    # handle the tuple case
                    if len(a_action) > 1:
                        if isinstance(a_action[0], np.ndarray):
                            a_action[0] = a_action[0].flatten()
                    action_dict[agent_id] = a_action
                    logits_dict[agent_id] = logits
                    prev_action = _flatten_action(a_action)  # tuple actions
                    prev_actions[agent_id] = prev_action

                    # Now store the agent action in the corresponding grid
                    if agent_id != 'robot':
                        indices = [0, 0]
                        bins = adversary_grid_dict[agent_id]['bins']
                        heat_map = adversary_grid_dict[agent_id]['grid']
                        for i, action in enumerate(a_action[0:2]):
                            bin_index = np.digitize(action, bins[i, :]) - 1
                            indices[i] = bin_index
                        heat_map[indices[0], indices[1]] += 1

            for agent_id in multi_obs.keys():
                if agent_id != 'robot':
                    # Now iterate through the agents and compute the kl_diff

                    curr_id = int(agent_id.split('adversary')[1])
                    your_logits = logits_dict[agent_id]
                    mean, log_std = np.split(your_logits.numpy()[0], 2)
                    for i in range(num_adversaries):
                        # KL diff of something with itself is zero
                        if i == curr_id:
                            pass
                        # otherwise just compute the kl difference between the agents
                        else:
                            other_logits = logits_dict['adversary{}'.format(i)]
                            other_mean, other_log_std = np.split(other_logits.numpy()[0], 2)
                            kl_diff = compute_kl_diff(log_std, mean, other_log_std, other_mean)
                            kl_grid[curr_id, i] += kl_diff

            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]

            # we turn the adversaries off so you only send in the robot keys
            new_dict = {}
            new_dict.update({'robot': action['robot']})
            next_obs, reward, done, info = env.step(new_dict)
            if isinstance(done, dict):
                done = done['__all__']
            step_num += 1
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            # we only want the robot reward, not the adversary reward
            reward_total += info['robot']['robot_reward']
            obs = next_obs
        total_steps += step_num

    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # Plot the heatmap of the actions
    for adversary, adv_dict in adversary_grid_dict.items():
        heat_map = adv_dict['grid']
        bins = adv_dict['bins']
        x_label, y_label = env.transform_adversary_actions(bins)
        # ax = sns.heatmap(heat_map, annot=True, fmt="d")
        plt.figure()
        sns.heatmap(heat_map, xticklabels=np.round(x_label, 2), yticklabels=np.round(y_label, 2))
        output_str = '{}/{}'.format(outdir, adversary + 'action_heatmap.png')
        plt.savefig(output_str)

    # Plot the kl difference between agents
    plt.figure()
    sns.heatmap(kl_grid / total_steps)
    output_str = '{}/{}'.format(outdir, 'kl_heatmap.png')
    plt.savefig(output_str)



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