import collections
import logging

import numpy as np
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

from envs.lerrel.adv_hopper import AdvMAHopper
from envs.lerrel.adv_inverted_pendulum_env import AdvMAPendulumEnv

from utils.pendulum_env_creator import pendulum_env_creator, lerrel_pendulum_env_creator, make_create_env

from models.conv_lstm import ConvLSTM
from models.recurrent_tf_model_v2 import LSTM

ModelCatalog.register_custom_model("rnn", ConvLSTM)
ModelCatalog.register_custom_model("rnn", LSTM)

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def instantiate_rollout(rllib_config, checkpoint):
    rllib_config['num_workers'] = 0

    # Determine agent and checkpoint
    assert rllib_config['env_config']['run'], "No RL algorithm specified in env config!"
    agent_cls = get_agent_class(rllib_config['env_config']['run'])
    # configure the env

    if rllib_config['env'] == "MALerrelPendulumEnv":
        env_name = "MALerrelPendulumEnv"
        create_env_fn = make_create_env(AdvMAPendulumEnv)
    elif rllib_config['env'] == "MALerrelHopperEnv":
        env_name = "MALerrelHopperEnv"
        create_env_fn = make_create_env(AdvMAHopper)

    register_env(env_name, create_env_fn)

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
        state_init = {}
        action_init = {}

    # We always have to remake the env since we may want to overwrite the config
    env = create_env_fn(rllib_config['env_config'])

    return env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init


def run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init, num_rollouts, render):

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
                        prev_action = _flatten_action(prev_actions[agent_id])
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_action,
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
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
                    prev_action = _flatten_action(a_action)  # tuple actions
                    prev_actions[agent_id] = prev_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]

            # we turn the adversaries off so you only send in the pendulum keys
            new_dict = {}
            new_dict.update({'agent': action['agent']})
            next_obs, reward, done, info = env.step(new_dict)
            new_dict.update({'agent': action['agent']})
            # next_obs, reward, done, info = env.step(action)
            if render:
                env.render()
            if isinstance(done, dict):
                done = done['__all__']
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            # we only want the robot reward, not the adversary reward
            reward_total += info['agent']['agent_reward']
            obs = next_obs
        print("Episode reward", reward_total)

        rewards.append(reward_total)

    env.close()

    print('the average reward is ', np.mean(rewards))
    return rewards