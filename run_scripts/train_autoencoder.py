"""This runs the env, gets a bunch of random samples and then saves the images to a folder.
    It then trains an autoencoder on that data and saves it for later reuse."""

# Step 1, rollout the env and save the images to a folder

import argparse
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.VAE import ConvVAE
from utils.env_creator import env_creator, construct_config
from utils.parsers import env_parser, init_parser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # can just override for multi-gpu systems
AUTOTUNE = tf.data.experimental.AUTOTUNE


def gather_images(passed_config, output_folder, horizon, total_num_steps):
    env = env_creator(passed_config)

    step_counter = 0
    ob = env.reset()
    while step_counter < total_num_steps:
        for i in range(horizon):
            assert np.all(ob > -1)
            assert np.all(ob < 1)
            matplotlib.image.imsave(os.path.join(output_folder, 'env_{}.jpg'.format(step_counter)),
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
#
#
def count_length_of_filelist(filelist):
    # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
    N = len(filelist)
    total_length = 0
    for i in range(N):
        filename = filelist[i]
        raw_data = np.load(os.path.join("record", filename))['obs']
        l = len(raw_data)
        total_length += l
        if (i % 1000 == 0):
            print("loading file", i)
    return total_length


def create_dataset(filepath, filelist, N=10000):  # N is 10000 episodes, M is number of timesteps
    length = min(len(filelist), N)
    data = np.zeros((length, 64, 64, 3), dtype=np.uint8)
    idx = 0
    for i in range(length):
        filename = filelist[i]
        # TODO(@evinitsky) we dont want the alpha channel in the fist place
        raw_data = plt.imread(os.path.join(filepath, filename), format='jpg')[:,:,0:3]
        data[idx] = raw_data
        idx += 1
        if ((i + 1) % 100 == 0):
            print("loading file", i + 1)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=500, help='How long each rollout should be')
    parser.add_argument('--total_step_num', type=int, default=50000,
                        help='How many total steps to collect for the autoencoder training')
    parser.add_argument('--output_folder', type=str, default='~/sim2real')
    parser.add_argument('--gather_images', default=False, action='store_true', help='Whether to gather images or just train')
    args = parser.parse_args()
    config = setup_sampling_env(parser)

    # Now construct the folder where we are saving images
    filepath = os.path.expanduser(args.output_folder)
    images_path = os.path.join(os.path.expanduser((filepath)), 'autoencoder_images')
    autoencoder_path = os.path.join(os.path.expanduser((filepath)), 'tf_vae')
    autoencoder_out = os.path.join(os.path.expanduser((filepath)), 'reconstructed_images')

    if not os.path.exists(os.path.expanduser(filepath)):
        os.makedirs(images_path)
        os.makedirs(autoencoder_path)
        os.makedirs(autoencoder_out)

    if len(os.listdir(filepath)) == 0 and not args.gather_images:
        sys.exit('You dont have any images in the training folder, so you should probably gather some. '
                 'Set --gather_images')

    if args.gather_images:
        gather_images(config, images_path, args.horizon, args.total_step_num)


    # Now lets train the auto-encoder
    np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

    # Hyperparameters for ConvVAE
    z_size = 32
    batch_size = 100
    learning_rate = 0.001
    kl_tolerance = 0.5

    # Parameters for training
    NUM_EPOCH = 500
    DATA_DIR = "record"

    # TODO(@evinitsky) maybe don't load the whole dataset into memory wtf
    # load dataset from record/*. only use first 10K, sorted by filename.
    filelist = os.listdir(images_path)
    filelist.sort()
    filelist = filelist[0:10000]
    # print("check total number of images:", count_length_of_filelist(filelist))
    dataset = create_dataset(images_path, filelist)

    # split into batches:
    total_length = len(dataset)
    num_batches = int(np.floor(total_length / batch_size))
    print("num_batches", num_batches)

    num_batches = int(np.floor(len(dataset) / batch_size))

    vae = ConvVAE(z_size=z_size,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  kl_tolerance=kl_tolerance,
                  is_training=True,
                  reuse=False,
                  gpu_mode=False)

    # train loop:
    # TODO(Stick this into tune and parallelize)
    print("train", "step", "loss", "recon_loss", "kl_loss")
    for epoch in range(NUM_EPOCH):
        np.random.shuffle(dataset)
        for idx in range(num_batches):
            batch = dataset[idx * batch_size:(idx + 1) * batch_size]
            obs = batch.astype(np.float) / 255.0

            feed = {vae.x: obs, }

            (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
                vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
            ], feed)

            if ((train_step + 1) % 10 == 0):
                print("step", (train_step + 1), train_loss, r_loss, kl_loss)

                # save a few images just for funsies
                # TODO(Since you're only extracting one image only save one image!)
                img = vae.sess.run([vae.y], feed)
                matplotlib.image.imsave(os.path.join(autoencoder_out, 'img_{}.png'.format(train_step)), img[0][0])
            if ((train_step + 1) % 5000 == 0):
                vae.save_json("tf_vae/vae.json")

    # finished, final model:
    vae.save_json(os.path.join(autoencoder_path, "vae.json"))
