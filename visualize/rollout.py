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
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot
from envs.policy.orca import ORCA
from utils.env_creator import env_creator

from run_scripts.test_rllib_script import env_creator

from models.models import ConvLSTM

ModelCatalog.register_custom_model("rnn", ConvLSTM)

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

def run_rollout(rllib_config, checkpoint, save_trajectory, video_file):
    rllib_config['num_workers'] = 0

    # Determine agent and checkpoint
    assert rllib_config['env_config']['run'], "No RL algorithm specified in env config!"
    agent_cls = get_agent_class(rllib_config['env_config']['run'])
    # configure the env
    env_name ='CrowdSim-v0'
    register_env(env_name, env_creator)

    # Instantiate the agent
    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=rllib_config)
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
        env = env_creator(rllib_config['env_config'])
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}


    # actually do the rollout
    mapping_cache = {}  # in case policy_agent_mapping is stochastic
    agent_states = DefaultMapping(
        lambda agent_id: state_init[mapping_cache[agent_id]])
    prev_actions = DefaultMapping(
        lambda agent_id: action_init[mapping_cache[agent_id]])
    obs = env.reset()
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
        obs = next_obs
        print("Episode reward", reward_total)

    if save_trajectory:
        env.render('traj', video_file)
    else:
        output_path = video_file
        if not output_path[-4:] == '.mp4':
            output_path += '.mp4'
        env.render('video', output_path)

    logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    if env.robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--video_file', type=str, default="rollout.mp4")
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

    checkpoint = args.result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    ray.init(num_cpus=args.num_cpus)

    run_rollout(rllib_config, checkpoint, args.traj, args.video_file)


if __name__ == '__main__':
    main()