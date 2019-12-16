import argparse
import configparser
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
    base_rewards, num_steps_base = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="base",
                               show_images=show_images, num_rollouts=num_rollouts)
    output_file_path = os.path.join(file_path, '{}/{}_base.txt'.format(outdir, output_file_name))
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(os.path.join(file_path, '{}/{}_base_rew.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, base_rewards, delimiter=', ')
    with open(os.path.join(file_path, '{}/{}_base_steps.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, num_steps_base, delimiter=', ')

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
    friction_rewards, num_steps_friction = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="friction",
                                   show_images=show_images, num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_friction_rew.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, friction_rewards, delimiter=', ')
    with open(os.path.join(file_path, '{}/{}_friction_steps.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, num_steps_friction, delimiter=', ')

    # # Now change the colors
    # print(
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "Running the colors score!\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************"
    # )
    # temp_config = reset_config(rllib_config)
    # temp_config['env_config']['change_colors_mode'] = 'every_step'
    # color_rewards, num_steps_colors = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="colours",
    #                             show_images=show_images,  num_rollouts=num_rollouts)
    # with open(os.path.join(file_path, '{}/{}_color_rew.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, color_rewards, delimiter=', ')
    # with open(os.path.join(file_path, '{}/{}_color_steps.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, num_steps_colors, delimiter=', ')
    #
    # # Allow goals to spawn anywhere (no longer restricted to smaller space)
    # print(
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "Running the changing goals score!\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************"
    # )
    #
    # # undo the goal restriction
    # temp_config = reset_config(rllib_config)
    # temp_config['env_config']['restrict_goal_region'] = False
    #
    # # enable goal randomization. We need to do it in this complicated way because
    # # this param is in the env_params.config instead of the env config
    # # TODO(@evinitsky) make this less gross
    # params_string = temp_config['env_config']['env_params']
    # params_parser = configparser.RawConfigParser()
    # params_parser.read_string(params_string)
    # params_parser['sim']['randomize_goals'] = str(True)
    # with open('tmp.config', 'w') as file:
    #     params_parser.write(file)
    # with open('tmp.config', 'r') as file:
    #     env_params = file.read()
    # os.remove('tmp.config')
    # temp_config['env_config']['env_params'] = env_params
    # unrestrict_goal_reg_rewards, num_steps_unrestrict = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="goal_reg",
    #                                           show_images=show_images, num_rollouts=num_rollouts)
    # with open(os.path.join(file_path, '{}/{}_unrestrict_goal_reg_rew.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, unrestrict_goal_reg_rewards, delimiter=', ')
    # with open(os.path.join(file_path, '{}/{}_unrestrict_goal_reg_steps.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, num_steps_unrestrict, delimiter=', ')
    #
    # # Have humans chase robot (hand-tuned adversarial human behaviour"
    # print(
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "Running the humans chasing agent score!\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************"
    # )
    # temp_config = reset_config(rllib_config)
    # temp_config['env_config']['chase_robot'] = True
    # chase_robot_rewards, num_steps_chase = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="chase_rob",
    #                                   show_images=show_images, num_rollouts=num_rollouts)
    # with open(os.path.join(file_path, '{}/{}_chase_robot_rew.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, chase_robot_rewards, delimiter=', ')
    # with open(os.path.join(file_path, '{}/{}_chase_robot_steps.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, num_steps_chase, delimiter=', ')
    #
    # # Add gaussian noise to the state
    # print(
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "Running the gaussian state noise score!\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************"
    # )
    # temp_config = reset_config(rllib_config)
    # temp_config['env_config']['add_gaussian_noise_state'] = True
    # gaussian_state_rewards, num_steps_GaS = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="gauss_state",
    #                                      show_images=show_images, num_rollouts=num_rollouts)
    # with open(os.path.join(file_path, '{}/{}_gaussian_state_noise_rew.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, gaussian_state_rewards, delimiter=', ')
    # with open(os.path.join(file_path, '{}/{}_gaussian_state_noise_steps.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, num_steps_GaS, delimiter=', ')
    #
    # # Add gaussian noise to the actions
    # print(
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "Running the gaussian action noise score!\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************"
    # )
    # temp_config = reset_config(rllib_config)
    # temp_config['env_config']['add_gaussian_noise_action'] = True
    # gaussian_action_rewards, num_steps_GaA = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="gauss_ac",
    #                                       show_images=show_images, num_rollouts=num_rollouts)
    # with open(os.path.join(file_path, '{}/{}_gaussian_action_noise_rew.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, gaussian_action_rewards, delimiter=', ')
    # with open(os.path.join(file_path, '{}/{}_gaussian_action_noise_steps.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, num_steps_GaA, delimiter=', ')
    #
    # # Add gaussian noise to the states and actions
    # print(
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "Running the gaussian state and action noise score!\n"
    #     "**********************************************************\n"
    #     "**********************************************************\n"
    #     "**********************************************************"
    # )
    # temp_config = reset_config(rllib_config)
    # temp_config['env_config']['add_gaussian_noise_action'] = True
    # temp_config['env_config']['add_gaussian_noise_action'] = True
    # gaussian_state_action_rewards, num_steps_GaA_GaS = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory, video_file="gauss_both",
    #                                             show_images=show_images, num_rollouts=num_rollouts)
    # with open(os.path.join(file_path, '{}/{}_gaussian_state_action_noise_rew.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, gaussian_state_action_rewards, delimiter=', ')
    # with open(os.path.join(file_path, '{}/{}_gaussian_state_action_noise_steps.txt'.format(outdir, output_file_name)),
    #           'wb') as file:
    #     np.savetxt(file, num_steps_GaA_GaS, delimiter=', ')
    # #
    print('The average base reward, episode length is {}, {}'.format(np.mean(base_rewards), np.mean(num_steps_base)))
    print('The average friction reward, episode length is {}, {}'.format(np.mean(friction_rewards), np.mean(num_steps_friction)))
    # print('The average color reward, episode length is {}, {}'.format(np.mean(color_rewards), np.mean(num_steps_colors)))
    # print('The average unrestricted goal region reward, episode length is {}, {}'.format(np.mean(unrestrict_goal_reg_rewards),
    #                                                                                      np.mean(num_steps_unrestrict)))
    # print('The average chasing robot reward, episode length is {}, {}'.format(np.mean(chase_robot_rewards), num_steps_chase))
    # print('The average Gaussian state reward, episode length is {}, {}'.format(np.mean(gaussian_state_rewards), num_steps_GaS))
    # print('The average Gaussian action reward, episode length is {}, {}'.format(np.mean(gaussian_action_rewards), num_steps_GaA))
    # print('The average Gaussian state + action reward, episode length is {}, {}'.format(np.mean(gaussian_state_action_rewards), num_steps_GaA_GaS))




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

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name, args.output_dir,
                       save_trajectory=args.save_video, show_images=args.show_images)