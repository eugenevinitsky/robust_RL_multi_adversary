import gym
from envs.trajectory_env import TrajectoryEnv
import configparser
import sys
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot

def configure_policy(passed_config, env, robot1, robot2, robot3, robot4):
    # configure policy
    policy_params = configparser.RawConfigParser()
    policy_params.read_string(passed_config['policy_params'])
    policy = policy_factory[passed_config['policy']](policy_params)
    if not policy.trainable:
        sys.exit('Policy has to be trainable')
    if passed_config['policy_params'] is None:
        sys.exit('Policy config has to be specified for a trainable network')

    robot1.set_policy(policy)
    robot2.set_policy(policy)
    robot3.set_policy(policy)
    robot4.set_policy(policy)

    policy.set_env(env)


def ma_env_creator(passed_config):
    config_path = passed_config['env_params']

    env_params = configparser.RawConfigParser()
    env_params.read_string(config_path)

    robot1 = Robot(env_params, 'robot', id=1)
    robot2 = Robot(env_params, 'robot', id=2)
    robot3 = Robot(env_params, 'robot', id=3)
    robot4 = Robot(env_params, 'robot', id=4)

    env = TrajectoryEnv(env_params, robot1, robot2, robot3, robot4)

    # control whether the last prediction element is continuous or discrete
    if passed_config['run'] == 'PPO':
        env.boxed_prediction = False
    elif passed_config['run'] == 'DDPG':
        env.boxed_prediction = True

    #configure_policy(passed_config, env, robot1, robot2, robot3, robot4)


    return env



def construct_config(env_params, policy_params, args):
    passed_config = {'env_params': env_params, 'policy_params': policy_params,
                     'policy': args.policy, 'show_images': args.show_images}
    return passed_config