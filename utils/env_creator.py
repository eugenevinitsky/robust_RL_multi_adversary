import configparser
import sys

from envs.crowd_env import CrowdSimEnv
from envs.policy.policy_factory import policy_factory
from envs.utils.robot import Robot


def env_creator(passed_config):
    config_path = passed_config['env_params']

    env_params = configparser.RawConfigParser()
    env_params.read_string(config_path)

    # additional configuration
    env_params['train_details']['show_images'] = str(passed_config['show_images'])
    env_params['train_details']['train_on_images'] = str(passed_config['train_on_images'])

    env_params['transfer']['change_colors_mode'] = str(passed_config['change_colors_mode'])
    env_params['transfer']['restrict_goal_region'] = str(passed_config['restrict_goal_region'])
    env_params['transfer']['chase_robot']  = str(passed_config['chase_robot'])
    env_params['transfer']['friction'] = str(passed_config['friction'])
    env_params['transfer']['friction_coef'] = str(passed_config['friction_coef'])

    env_params['env']['add_gaussian_noise_state'] = str(passed_config['add_gaussian_noise_state'])
    env_params['env']['add_gaussian_noise_action'] = str(passed_config['add_gaussian_noise_action'])

    robot = Robot(env_params, 'robot')
    env = CrowdSimEnv(env_params, robot)

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
                     'train_on_images': args.train_on_images, 'friction': args.friction,
                     'friction_coef': args.friction_coef, 'chase_robot': args.chase_robot,
                     'restrict_goal_region': args.restrict_goal_region,
                     'add_gaussian_noise_state': args.add_gaussian_noise_state,
                     'add_gaussian_noise_action': args.add_gaussian_noise_action,}
    return passed_config