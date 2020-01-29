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


def run_rollout(rllib_config, checkpoint, num_rollouts, show_image):
    rllib_config['num_envs_per_worker'] = 1
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)
    if show_image:
        env.show_image = True

    mapping_cache = {}  # in case policy_agent_mapping is stochastic

    sample_idx = 0
    while sample_idx < num_rollouts:
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        if env.adversary_range > 0:
            env.curr_adversary = np.random.randint(low=0, high=env.adversary_range)
        print('on rollout {}'.format(sample_idx))
        obs = env.reset()
        action_dict = {}
        # we have an is_active key here
        # multi_obs = {'agent': obs}
        done = {}
        rew = 0
        done['__all__'] = False
        while not done['__all__']:
            for agent_id, a_obs in obs.items():
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
                    prev_actions[agent_id] = a_action
                    action_dict[agent_id] = a_action
            obs, reward, done, info = env.step(action_dict)
            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r
            rew += reward['agent']
        print('the reward of rollout {} was {}'.format(sample_idx, rew))
        sample_idx += 1

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--output_dir', type=str, default='transfer_results/linear_env',
                        help='Directory to output the files into')
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    rllib_config, checkpoint = get_config(args)
    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})

    ray.init(num_cpus=args.num_cpus)

    run_rollout(rllib_config, checkpoint, args.num_rollouts, True)

if __name__ == '__main__':
    main()