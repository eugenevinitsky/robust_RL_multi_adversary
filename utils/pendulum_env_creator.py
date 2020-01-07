import gym
from envs.inverted_pendulum_env import MAPendulumEnv, PendulumEnv, ModelBasedPendulumEnv

def pendulum_env_creator(env_config):
    if env_config['num_adversaries'] > 0:
        if env_config['model_based']:
            env = ModelBasedPendulumEnv(env_config)
        else:
            env = MAPendulumEnv(env_config)
    else:
        env = PendulumEnv()
    return env
