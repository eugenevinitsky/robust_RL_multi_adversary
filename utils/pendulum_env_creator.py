import gym
from envs.inverted_pendulum_env import MAPendulumEnv, PendulumEnv
from envs.lerrel.adv_inverted_pendulum_env import AdvMAPendulumEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv

def pendulum_env_creator(env_config):
    if env_config['num_adversaries'] > 0:
        env = MAPendulumEnv(env_config)
    else:
        env = PendulumEnv()
    return env

def lerrel_pendulum_env_creator(env_config):
    if env_config['num_adversaries'] > 0:
        env = AdvMAPendulumEnv(env_config)
        # env = InvertedPendulumEnv()
    else:
        env = PendulumEnv()
    return env
