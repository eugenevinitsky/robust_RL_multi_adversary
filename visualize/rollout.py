import argparse
import collections
import configparser
import logging
import os
import sys

import gym
import numpy as np
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.cloudpickle import cloudpickle
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.tune.registry import register_env

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot
from envs.policy.orca import ORCA


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def create_env(config):
    policy_config = configparser.RawConfigParser()
    policy_config.read_string(config['policy_config'])
    policy = policy_factory[config['policy']](policy_config)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read_string(config['env_config'])

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env = CrowdSimEnv(env_config)
    if config['replay_params'].get('square', False):
        env.test_sim = 'square_crossing'
    if config['replay_params'].get('circle', False):
        env.test_sim = 'circle_crossing'


    policy.set_phase(config['replay_params']['phase'])
    # set safety space for ORCA in non-cooperative simulation
    # TODO(@evinitsky) wtf is this
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    return env


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--run', type=str, help='RL algorithm that is run')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    config_path = os.path.join(args.result_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.result_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    with open(config_path, 'rb') as f:
        rllib_config = cloudpickle.load(f)

    rllib_config['num_workers'] = 0

    # Determine agent and checkpoint
    config_run = rllib_config['env_config']['run'] if 'run' in rllib_config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    # configure the env
    # TOO @(evinitsky) overwrite replay params with arg params
    env_config = rllib_config['env_config']['replay_params']
    if args.circle:
        env_config['circle'] = True
    if args.square:
        env_config['square'] = True
    env_config['phase'] = args.phase
    env_name ='CrowdSim-v0'
    register_env(env_name, create_env)

    # Instantiate the agent
    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=rllib_config)
    checkpoint = args.result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    ray.init(num_cpus=args.num_cpus)
    agent.restore(checkpoint)

    policy_agent_mapping = default_policy_agent_mapping
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = create_env(rllib_config['env_config'])
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}


    # actually do the rollout
    if args.traj is not None:
        rollout = []
    mapping_cache = {}  # in case policy_agent_mapping is stochastic
    agent_states = DefaultMapping(
        lambda agent_id: state_init[mapping_cache[agent_id]])
    prev_actions = DefaultMapping(
        lambda agent_id: action_init[mapping_cache[agent_id]])
    obs = env.reset(args.phase, args.test_case)
    prev_rewards = collections.defaultdict(lambda: 0.)
    done = False
    reward_total = 0.0
    while not done:
        multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
        action_dict = {}
        for agent_id, a_obs in multi_obs.items():
            if a_obs is not None:
                policy_id = mapping_cache.setdefault(
                    agent_id, policy_agent_mapping(agent_id))
                p_use_lstm = use_lstm[policy_id]
                if p_use_lstm:
                    a_action, p_state, _ = agent.compute_action(
                        a_obs,
                        state=agent_states[agent_id],
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                    agent_states[agent_id] = p_state
                else:
                    a_action = agent.compute_action(
                        a_obs,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id)
                a_action = _flatten_action(a_action)  # tuple actions
                action_dict[agent_id] = a_action
                prev_actions[agent_id] = a_action
        action = action_dict

        action = action if multiagent else action[_DUMMY_AGENT_ID]
        next_obs, reward, done, info = env.step(action)
        if multiagent:
            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r
        else:
            prev_rewards[_DUMMY_AGENT_ID] = reward

        if multiagent:
            done = done["__all__"]
            reward_total += sum(reward.values())
        else:
            reward_total += reward
        if args.traj is not None:
            rollout.append([obs, action, next_obs, reward, done])
        obs = next_obs
        print("Episode reward", reward_total)

    if args.traj:
        env.render('traj', args.video_file)
    else:
        output_path = args.video_file
        if not output_path[-4:] == '.mp4':
            output_path += '.mp4'
        env.render('video', output_path)

    logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    if env.robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))



if __name__ == '__main__':
    main()