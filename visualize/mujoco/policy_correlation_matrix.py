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

def visualize_adversaries(config_out_dir, checkpoint_num, grid_size, num_rollouts, outdir, plot_base_case,
                          extension):

    agent_list = []
    index = 0
    # max_index = 20000
    multiagent = True
    for (dirpath, dirnames, filenames) in os.walk(config_out_dir):
        if "params.pkl" in filenames:
            # if index > max_index:
            #     break
            rllib_config, checkpoint = get_config_from_path(dirpath, checkpoint_num)
            env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
                instantiate_rollout(rllib_config, checkpoint)
            agent_list.append(agent)
            # index += 1

    # figure out how many adversaries you have and initialize their grids
    num_adversaries = env.num_adv_strengths * env.advs_per_strength
    if plot_base_case:
        policy_correlation_grid = np.zeros((len(agent_list), len(agent_list) + 1))
    else:
        policy_correlation_grid = np.zeros((len(agent_list), len(agent_list)))

    if plot_base_case:
        adversary_loop_index = len(agent_list) + 1
    else:
        adversary_loop_index = len(agent_list)

    for agent_row_index in range(len(agent_list)):
        print('Outer index {}'.format(agent_row_index))
        for adversary_col_index in range(adversary_loop_index):
            print('Inner index {}'.format(adversary_col_index))
            reward_total = 0.0
            for adversary_index in range(num_adversaries):
                env.curr_adversary = adversary_index
                # turn the adversaries off for the last column
                if adversary_col_index == len(agent_list):
                    env.curr_adversary = -1
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
                        new_obs = {'agent': multi_obs['agent']}
                        # turn the adversaries off for the last column
                        if adversary_col_index < len(agent_list):
                            new_obs.update({'adversary{}'.format(adversary_index): multi_obs['adversary{}'.format(adversary_index)]})
                        action_dict = {}
                        for agent_id, a_obs in new_obs.items():
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

                        next_obs, reward, done, info = env.step(action)
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
    plt.imshow(policy_correlation_grid, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
    plt.colorbar()
    fontsize = 14
    title_fontsize = 16
    plt.title('Policy Correlation Matrix', fontsize=title_fontsize)
    if plot_base_case:
        plt.yticks(ticks=np.arange(len(agent_list)))
        plt.xticks(ticks=np.arange(len(agent_list)).tolist().append(['base']))
    else:
        plt.yticks(ticks=np.arange(len(agent_list)))
        plt.xticks(ticks=np.arange(len(agent_list)))
    plt.ylabel('agent index', fontsize=fontsize)
    plt.xlabel('adversary index', fontsize=fontsize)
    output_str = '{}/{}'.format(os.path.abspath(os.path.expanduser(outdir)), 'policy_correlation_map_{}.png'.format(extension))
    with open('{}/{}'.format(os.path.abspath(os.path.expanduser(outdir)), 'results_{}'.format(extension)),
              'wb') as file:
        np.savetxt(file, policy_correlation_grid)
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(output_str)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--grid_size', type=int, default=50, help='How fine to make the adversary action grid')
    parser.add_argument('--output_dir', type=str, default='~/transfer_results',
                        help='Directory to output the files into')
    parser.add_argument('--plot_base_case', action='store_true', default=False,
                        help='Add an additional column where no adversary is on')
    parser.add_argument('--extension', type=str, default='')
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    ray.init(num_cpus=args.num_cpus)

    visualize_adversaries(args.result_dir, args.checkpoint_num, args.grid_size, args.num_rollouts, args.output_dir,
                          args.plot_base_case, args.extension)

if __name__ == '__main__':
    main()