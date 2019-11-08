"""This runs the env, gets a bunch of random samples and then saves the images to a folder.
    It then trains an autoencoder on that data and saves it for later reuse."""

# Step 1, rollout the env and save the images to a folder

import argparse
import configparser
import os
import sys

import matplotlib
import numpy as np

from utils.env_creator import env_creator, construct_config
from utils.parsers import env_parser, init_parser


def gather_images(passed_config, output_folder, horizon, total_num_steps):
    env = env_creator(passed_config)

    step_counter = 0
    ob = env.reset()
    while step_counter < total_num_steps:
        for i in range(horizon):
            assert np.all(ob > -1)
            assert np.all(ob < 1)
            matplotlib.image.imsave(os.path.join(output_folder, 'env_{}.png'.format(step_counter)),
                                    ob[:, :, 0:3] + (128/255))
            # write the image to a file
            ob, rew, done, info = env.step(np.random.rand(2))
            if done:
                ob = env.reset()
        step_counter += 1


def setup_sampling_env(parser):
    args = env_parser(parser).parse_args()
    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()

    passed_config = construct_config(env_params, policy_params, args)
    # We need images so that we can actually store them
    passed_config['train_on_images'] = True

    return passed_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=500, help='How long each rollout should be')
    parser.add_argument('--total_step_num', type=int, default=50000,
                        help='How many total steps to collect for the autoencoder training')
    parser.add_argument('--output_folder', type=str, default='~/sim2real/autoencoder_images')
    parser.add_argument('--gather_images', default=False, action='store_true', help='Whether to gather images or just train')
    args = parser.parse_args()
    config = setup_sampling_env(parser)

    # Now construct the folder where we are saving images
    filepath = os.path.expanduser(args.output_folder)
    if not os.path.exists(os.path.expanduser(filepath)):
        os.makedirs(os.path.expanduser((filepath)))

    if len(os.listdir(filepath)) == 0 and not args.gather_images:
        sys.exit('You dont have any images in the training folder, so you should probably gather some. '
                 'Set --gather_images')

    if args.gather_images:
        gather_images(config, filepath, args.horizon, args.total_step_num)

    # Now lets train the auto-encoder
