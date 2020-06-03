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

from visualize.mujoco.run_rollout import instantiate_rollout, DefaultMapping
from utils.parsers import replay_parser
from utils.rllib_utils import get_config_from_path

# TODO(@evinitsky) make this work for the case where you have multiple adversaries
def visualize_adversaries(config_out_dir, checkpoint_num, grid_size, num_rollouts, outdir):

    agent_list = []
    for (dirpath, dirnames, filenames) in os.walk(config_out_dir):
        if "params.pkl" in filenames:
            rllib_config, checkpoint = get_config_from_path(dirpath, checkpoint_num)
            env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
                instantiate_rollout(rllib_config, checkpoint)
            agent_list.append(agent)

    # figure out how many adversaries you have and initialize their grids
    num_adversaries = env.num_adv_strengths * env.advs_per_strength
    policy_correlation_grid = np.zeros((len(agent_list), len(agent_list)))

    for agent_row_index in range(len(agent_list)):
        for adversary_col_index in range(len(agent_list)):
            reward_total = 0.0
            for adversary_index in range(num_adversaries):
                # actually do the rollouts
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
                    step_num = 0
                    while not done:
                        multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
                        obs = multi_obs['agent']
                        multi_obs = {'adversary{}'.format(adversary_index)}
                        multi_obs.update({'agent': obs})
                        action_dict = {}
                        for agent_id, a_obs in multi_obs.items():
                            if 'agent' in agent_id:
                                if a_obs is not None:
                                    policy_id = mapping_cache.setdefault(
                                        agent_id, policy_agent_mapping(agent_id))
                                    p_use_lstm = use_lstm[policy_id]
                                    if p_use_lstm:
                                        prev_action = _flatten_action(prev_actions[agent_id])
                                        a_action, p_state, _ = agent_list[agent_row_index].compute_action(
                                            a_obs,
                                            state=agent_states[agent_id],
                                            prev_action=prev_action,
                                            prev_reward=prev_rewards[agent_id],
                                            policy_id=policy_id)
                                        agent_states[agent_id] = p_state
                                    else:
                                        prev_action = _flatten_action(prev_actions[agent_id])
                                        flat_action = _flatten_action(a_obs)
                                        a_action = agent_list[agent_row_index].compute_action(
                                            flat_action,
                                            prev_action=prev_action,
                                            prev_reward=prev_rewards[agent_id],
                                            policy_id=policy_id)
                            else:
                                if a_obs is not None:
                                    policy_id = mapping_cache.setdefault(
                                        agent_id, policy_agent_mapping(agent_id))
                                    p_use_lstm = use_lstm[policy_id]
                                    if p_use_lstm:
                                        prev_action = _flatten_action(prev_actions[agent_id])
                                        a_action, p_state, _ = agent_list[adversary_col_index].compute_action(
                                            a_obs,
                                            state=agent_states[agent_id],
                                            prev_action=prev_action,
                                            prev_reward=prev_rewards[agent_id],
                                            policy_id=policy_id)
                                        agent_states[agent_id] = p_state
                                    else:
                                        prev_action = _flatten_action(prev_actions[agent_id])
                                        flat_action = _flatten_action(a_obs)
                                        a_action = agent_list[adversary_col_index].compute_action(
                                            flat_action,
                                            prev_action=prev_action,
                                            prev_reward=prev_rewards[agent_id],
                                            policy_id=policy_id)

                                action_dict[agent_id] = a_action
                                prev_action = _flatten_action(a_action)  # tuple actions
                                prev_actions[agent_id] = prev_action

                        action = action_dict

                        action = action if multiagent else action[_DUMMY_AGENT_ID]

                        new_dict = {}
                        new_dict.update({'agent': action['agent']})
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
                        reward_total += info['agent']['agent_reward']
                        obs = next_obs
            policy_correlation_grid[agent_row_index, adversary_col_index] = reward_total / (num_rollouts * num_adversaries)

    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # increasing the row index implies moving down on the y axis
    sns.heatmap(policy_correlation_grid)
    plt.ylabel('agent index')
    plt.xlabel('adversary index')
    output_str = '{}/{}'.format(outdir, 'policy_correlation_map.png')
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

    ray.init(num_cpus=args.num_cpus)

    visualize_adversaries(args.result_dir, args.checkpoint_num, args.grid_size, args.num_rollouts, args.output_dir)

if __name__ == '__main__':
    main()