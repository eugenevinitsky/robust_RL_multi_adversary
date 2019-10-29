import configparser
import sys

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot


def env_creator(passed_config):
    config_path = passed_config['env_params']

    env_params = configparser.RawConfigParser()
    env_params.read_string(config_path)

    robot = Robot(env_params, 'robot')
    env = CrowdSimEnv(env_params, robot)

    # additional configuration
    env.show_images = passed_config['show_images']
    env.train_on_images = passed_config['train_on_images']
    env.change_colors_mode = passed_config['change_colors_mode']

    # configure policy
    policy_params = configparser.RawConfigParser()
    policy_params.read_string(passed_config['policy_params'])
    policy = policy_factory[passed_config['policy']](policy_params)
    if not policy.trainable:
        sys.exit('Policy has to be trainable')
    if passed_config['policy_params'] is None:
        sys.exit('Policy config has to be specified for a trainable network')

    robot.set_policy(policy)
    policy.set_env(env)
    robot.print_info()
    return env


def construct_config(env_params, policy_params, args):
    passed_config = {'env_params': env_params, 'policy_params': policy_params,
                     'policy': args.policy, 'show_images': args.show_images,
                     'change_colors_mode': args.change_colors_mode,
                     'train_on_images': args.train_on_images, 'friction': args.friction}
    return passed_config
