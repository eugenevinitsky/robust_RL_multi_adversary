import argparse
from copy import deepcopy
import os

import numpy as np
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.rollout import run_rollout
import errno


def reset_config(rllib_config):
    """Return a copy of the config with the possible transfer parameters set to default values"""
    copy_config = deepcopy(rllib_config)
    copy_config['env_config']['friction'] = False
    copy_config['env_config']['change_colors_mode'] = 'no_change'
    copy_config['env_config']['restrict_goal_region'] = True
    copy_config['env_config']['chase_robot'] = False
    copy_config['env_config']['add_gaussian_noise_state'] = False
    copy_config['env_config']['add_gaussian_noise_action'] = False
    return copy_config


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir, save_trajectory, show_images=False):
    # TODO(@Evinitsky) when you run the transfer tests for multi-agent you want to run it with no adversaries!!!
    file_path = os.path.dirname(os.path.abspath(__file__))

    # First compute a baseline score to compare against
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the baseline score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    base_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="base",
                               show_images=show_images, num_rollouts=num_rollouts)
    output_file_path = os.path.join(file_path, '{}/{}_base.txt'.format(outdir, output_file_name))
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(os.path.join(file_path, '{}/{}_base.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, base_rewards, delimiter=', ')

    # First lets do the friction rewards
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the friction score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['friction'] = True
    friction_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="friction",
                                   show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_friction.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, friction_rewards, delimiter=', ')

    # Now change the colors
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the colors score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['change_colors_mode'] = 'every_step'
    color_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="colours",
                                show_images=show_images,  num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_color.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, color_rewards, delimiter=', ')

    # Allow goals to spawn anywhere (no longer restricted to smaller space)
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the changing goals score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['restrict_goal_region'] = False
    unrestrict_goal_reg_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="goal_reg",
                                              show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_unrestrict_goal_reg.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, unrestrict_goal_reg_rewards, delimiter=', ')

    # Have humans chase robot (hand-tuned adversarial human behaviour"
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the humans chasing agent score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['chase_robot'] = True
    chase_robot_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="chase_rob",
                                      show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_chase_robot.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, chase_robot_rewards, delimiter=', ')

    # Add gaussian noise to the state
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the gaussian state noise score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['add_gaussian_noise_state'] = True
    gaussian_state_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="gauss_state",
                                         show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_gaussian_state_noise.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, gaussian_state_rewards, delimiter=', ')

    # Add gaussian noise to the actions
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the gaussian action noise score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['add_gaussian_noise_action'] = True
    gaussian_action_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="gauss_ac",
                                          show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_gaussian_action_noise.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, gaussian_action_rewards, delimiter=', ')

    # Add gaussian noise to the states and actions
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the gaussian state and action noise score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************"
    )
    temp_config = reset_config(rllib_config)
    temp_config['env_config']['add_gaussian_noise_action'] = True
    temp_config['env_config']['add_gaussian_noise_action'] = True
    gaussian_state_action_rewards = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="gauss_both",
                                                show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_gaussian_state_action_noise.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, gaussian_state_action_rewards, delimiter=', ')

    print('The average base reward is {}'.format(np.mean(base_rewards)))
    print('The average friction reward is {}'.format(np.mean(friction_rewards)))
    print('The average color reward is {}'.format(np.mean(color_rewards)))
    print('The average unrestricted goal region reward is {}'.format(np.mean(unrestrict_goal_reg_rewards)))
    print('The average chasing robot reward is {}'.format(np.mean(chase_robot_rewards)))
    print('The average Gaussian state reward is {}'.format(np.mean(gaussian_state_rewards)))
    print('The average Gaussian action reward is {}'.format(np.mean(gaussian_action_rewards)))
    print('The average Gaussian state + action reward is {}'.format(np.mean(gaussian_state_action_rewards)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--output_dir', type=str, default='transfer_results',
                        help='')

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)
    save_trajectory = False
    if args.traj == 'video':
        save_trajectory = True
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name, args.output_dir,
                       save_trajectory=save_trajectory, show_images=args.show_images)