import collections
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

from utils.pendulum_env_creator import pendulum_env_creator

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


def instantiate_rollout(rllib_config, checkpoint, show_images):
    rllib_config['num_workers'] = 0

    # Determine agent and checkpoint
    assert rllib_config['env_config']['run'], "No RL algorithm specified in env config!"
    agent_cls = get_agent_class(rllib_config['env_config']['run'])
    # configure the env
    env_name ='MAPendulumEnv'
    register_env(env_name, pendulum_env_creator)

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
    env = pendulum_env_creator(rllib_config['env_config'])

    return env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init
