import argparse
from copy import deepcopy
import os

import numpy as np
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.rollout import run_rollout
import errno


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir='transfer_results'):
    file_path = os.path.dirname(os.path.abspath(__file__))

    # First compute a baseline score to compare against
    base_rewards = run_rollout(rllib_config, checkpoint, save_trajectory=False, video_file=False,
                               num_rollouts=num_rollouts)
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
    temp_config = deepcopy(rllib_config)
    temp_config['env_config']['friction'] = True
    friction_rewards = run_rollout(temp_config, checkpoint, save_trajectory=False, video_file=False,
                                   num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_friction.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, friction_rewards, delimiter=', ')

    # Now change the colors
    temp_config = deepcopy(rllib_config)
    temp_config['env_config']['change_colors_mode'] = 'every_step'
    color_rewards = run_rollout(temp_config, checkpoint, save_trajectory=False, video_file=False,
                                num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_color.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, friction_rewards, delimiter=', ')

    # Allow goals to spawn anywhere (no longer restricted to smaller space)
    temp_config = deepcopy(rllib_config)
    temp_config['env_config']['restrict_goal_region'] = False
    unrestrict_goal_reg_rewards = run_rollout(temp_config, checkpoint, save_trajectory=False, video_file=False,
                                num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_unrestrict_goal_reg.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, unrestrict_goal_reg_rewards, delimiter=', ')

    # Have humans chase robot (hand-tuned adversarial human behaviour"
    temp_config = deepcopy(rllib_config)
    temp_config['env_config']['chase_robot'] = True
    chase_robot_rewards = run_rollout(temp_config, checkpoint, save_trajectory=False, video_file=False,
                                              num_rollouts=num_rollouts)
    with open(os.path.join(file_path, '{}/{}_chase_robot.txt'.format(outdir, output_file_name)),
              'wb') as file:
        np.savetxt(file, chase_robot_rewards, delimiter=', ')

    print('The average base reward is {}'.format(np.mean(base_rewards)))
    print('The average friction reward is {}'.format(np.mean(friction_rewards)))
    print('The average color reward is {}'.format(np.mean(color_rewards)))
    print('The average unrestricted goal region reward is {}'.format(np.mean(unrestrict_goal_reg_rewards)))
    print('The average chasing robot reward is {}'.format(np.mean(chase_robot_rewards)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name)