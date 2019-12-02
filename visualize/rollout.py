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

from utils.env_creator import env_creator, ma_env_creator

from utils.parsers import replay_parser
from utils.rllib_utils import get_config

from models.conv_lstm import ConvLSTM

ModelCatalog.register_custom_model("rnn", ConvLSTM)

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def run_rollout(rllib_config, checkpoint, save_trajectory, video_file, show_images, num_rollouts):
    rllib_config['num_workers'] = 0

    # Determine agent and checkpoint
    assert rllib_config['env_config']['run'], "No RL algorithm specified in env config!"
    agent_cls = get_agent_class(rllib_config['env_config']['run'])
    # configure the env
    env_name ='CrowdSim-v0'
    if 'multiagent' in rllib_config and rllib_config['multiagent']['policies']:
        register_env(env_name, ma_env_creator)
    else:
        register_env(env_name, env_creator)

    # Show the images
    rllib_config['env_config']['show_images'] = show_images

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
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    # We always have to remake the env since we may want to overwrite the config
    if 'multiagent' in rllib_config and rllib_config['multiagent']['policies']:
        env = ma_env_creator(rllib_config['env_config'])
    else:
        env = env_creator(rllib_config['env_config'])

    rewards = []

    # actually do the rollout
    for r_itr in range(num_rollouts):
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

            # TODO(@evinitsky) make this a config option
            if multiagent:
                for key, value in action.items():
                    if key != 'robot':
                        action[key] = np.zeros(value.shape)
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                # TODO(@evinitsky) make this a config option
                reward_total += reward['robot']
            else:
                reward_total += reward
            obs = next_obs
        print("Episode reward", reward_total)
        rewards.append(reward_total)

    if not show_images:
        if save_trajectory:
            env.render('traj', video_file)
            output_path = video_file
            if not output_path[-4:] == '.mp4':
                output_path += '_.mp4'
            env.render('video', output_path)
    else:
        logging.info('Video creation is disabled since show_images is true.')


    logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    if env.robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

    return rewards
    

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    rllib_config, checkpoint = get_config(args)
    rllib_config['env_config']['chase_robot'] = True

    ray.init(num_cpus=args.num_cpus)

    run_rollout(rllib_config, checkpoint, args.traj, args.video_file, args.show_images, args.num_rollouts)


if __name__ == '__main__':
    main()