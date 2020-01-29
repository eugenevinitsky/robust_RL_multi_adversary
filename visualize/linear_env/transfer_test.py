import argparse
import collections
from datetime import datetime
import errno
import logging
import os

import numpy as np
import pytz
import ray
from ray.rllib.evaluation.episode import _flatten_action

from visualize.pendulum.run_rollout import instantiate_rollout, DefaultMapping
from utils.parsers import replay_parser
from utils.rllib_utils import get_config


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir):
    output_file_path = os.path.join(outdir, output_file_name)
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    rllib_config['num_envs_per_worker'] = 1
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    mapping_cache = {}  # in case policy_agent_mapping is stochastic

    # set the adversary range to zero so that we get domain randomization
    env.adversary_range = 0

    rew_list = []
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
        rew_list.append(rew)
        sample_idx += 1

    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, "mean_sweep"),
              'wb') as file:
        np.save(file, np.array(rew_list))

    rew_list = []
    sample_idx = 0
    while sample_idx < num_rollouts:
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        if env.adversary_range > 0:
            env.curr_adversary = np.random.randint(low=0, high=env.adversary_range)
        print('on rollout {}'.format(sample_idx))
        obs = env.reset()
        # turn off the perturbations to get a base score
        env.should_perturb = False
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
        rew_list.append(rew)
        sample_idx += 1

    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, "base_sweep"),
              'wb') as file:
        np.save(file, np.array(rew_list))

def main():
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    output_path = os.path.expanduser('~/transfer_results/linear_env')

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--output_dir', type=str, default=output_path,
                        help='')

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    ray.init(num_cpus=args.num_cpus)

    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name,
                       os.path.join(args.output_dir, date))
if __name__ == '__main__':
    main()