import gym
from envs.inverted_pendulum_env import MAPendulumEnv, PendulumEnv

def pendulum_env_creator(env_config):
    if env_config['num_adversaries'] > 0:
        env = MAPendulumEnv(env_config)
    else:
        env = PendulumEnv()
    return env
