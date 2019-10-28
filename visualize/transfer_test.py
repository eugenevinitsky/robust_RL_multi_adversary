import argparse
from copy import deepcopy
import os

import numpy as np
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.rollout import run_rollout

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)

    file_path = os.path.dirname(os.path.abspath(__file__))

    # First compute a baseline score to compare against
    base_rewards = run_rollout(rllib_config, checkpoint, save_trajectory=False, video_file=False,
                                   num_rollouts=args.num_rollouts)
    with open(os.path.join(file_path, 'transfer_results/{}_base.txt'.format(args.output_file_name)),
              'wb') as file:
        np.savetxt(file, base_rewards, delimiter=', ')

    # First lets do the friction rewards
    temp_config = deepcopy(rllib_config)
    temp_config['env_config']['friction'] = True
    friction_rewards = run_rollout(temp_config, checkpoint, save_trajectory=False, video_file=False,
                                   num_rollouts=args.num_rollouts)
    with open(os.path.join(file_path, 'transfer_results/{}_friction.txt'.format(args.output_file_name)),
              'wb') as file:
        np.savetxt(file, friction_rewards, delimiter=', ')

    # Now change the colors
    temp_config = deepcopy(rllib_config)
    temp_config['env_config']['change_colors_mode'] = 'every_step'
    color_rewards = run_rollout(temp_config, checkpoint, save_trajectory=False, video_file=False,
                                   num_rollouts=args.num_rollouts)
    with open(os.path.join(file_path, 'transfer_results/{}_color.txt'.format(args.output_file_name)),
              'wb') as file:
        np.savetxt(file, friction_rewards, delimiter=', ')

    print('The average base reward is {}'.format(np.mean(base_rewards)))
    print('The average friction reward is {}'.format(np.mean(friction_rewards)))
    print('The average color reward is {}'.format(np.mean(color_rewards)))
