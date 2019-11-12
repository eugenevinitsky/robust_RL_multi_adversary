"""This runs the env, gets a bunch of random samples and then saves the images to a folder.
    It then trains an autoencoder on that data and saves it for later reuse."""

# Step 1, rollout the env and save the images to a folder

import argparse
import os
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ray.tune import Trainable, run
from ray.tune.logger import DEFAULT_LOGGERS
import tensorflow as tf

from models.VAE import ConvVAE
from utils.custom_tf_logger import TFLoggerPlus
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


class ConvVaeTrainer(Trainable):
    def _setup(self, config):
        # Now construct the folder where we are saving images
        filepath = os.path.expanduser(config['output_folder'])
        images_path = os.path.join(os.path.expanduser((filepath)), 'autoencoder_images')
        autoencoder_path = os.path.join(os.path.expanduser((filepath)), 'tf_vae')
        autoencoder_out = os.path.join(os.path.expanduser((filepath)), 'reconstructed_images')

        if not os.path.exists(os.path.expanduser(filepath)):
            os.makedirs(images_path)
            os.makedirs(autoencoder_path)
            os.makedirs(autoencoder_out)

        if len(os.listdir(filepath)) == 0 and not config['gather_images']:
            sys.exit('You dont have any images in the training folder, so you should probably gather some. '
                     'Set --gather_images')

        if args.gather_images:
            gather_images(config, images_path, args.horizon, args.total_step_num)

        # Now lets train the auto-encoder
        np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

        # Hyperparameters for ConvVAE
        self.batch_size = config['batch_size']

        # Parameters for training
        self.NUM_EPOCH = 500

        # TODO(@evinitsky) maybe don't load the whole dataset into memory wtf
        # load dataset from record/*. only use first 10K, sorted by filename.
        filelist = os.listdir(images_path)
        filelist.sort()
        filelist = filelist[0:10000]
        # print("check total number of images:", count_length_of_filelist(filelist))
        self.dataset = create_dataset(images_path, filelist)

        # split into batches:
        total_length = len(self.dataset)
        self.num_batches = int(np.floor(total_length / self.batch_size))
        print("num_batches", self.num_batches)

        self.num_batches = int(np.floor(len(self.dataset) / self.batch_size))

        self.vae = ConvVAE(z_size=config['z_size'],
                          batch_size=config['batch_size'],
                          learning_rate=config['learning_rate'],
                          kl_tolerance=config['kl_tolerance'],
                          is_training=True,
                          reuse=False,
                          gpu_mode=config['use_gpu'])

    def _train(self):
        # train loop:
        np.random.shuffle(self.dataset)
        for idx in range(self.num_batches):
            batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
            obs = batch.astype(np.float) / 255.0

            feed = {self.vae.x: obs, }

            (train_loss, r_loss, kl_loss, train_step, grads_and_vars, activation_list, _) = self.vae.sess.run([
                self.vae.loss, self.vae.r_loss, self.vae.kl_loss, self.vae.global_step, self.vae.grads,
                self.vae.activation_list, self.vae.train_op
            ], feed)

            # TODO(@evinitsky) add gradient norm, add histogram of activations and weights

            results = {
                "epoch": self.iteration,
                "train_loss": train_loss,
                "r_loss": r_loss,
                "kl_loss": kl_loss,
            }

            grad_squares = [np.square(g).sum() for (g, v) in grads_and_vars]
            for i, grad_square in enumerate(grad_squares):
                if i % 2 == 0:
                    results.update({"layer_{}_grad_norm".format(i / 2): np.sqrt(grad_square)})
                else:
                    results.update({"bias_{}_grad_norm".format((i - 1) / 2): np.sqrt(grad_square)})

            vars = [v for (g, v) in grads_and_vars]
            weights = vars[::2]
            biases = vars[1::2]

            if ((train_step) % 10 == 0):
                # Save histograms of the activations
                # Okay, now lets make a hist of the activations and a hist of the weights
                for activation, activation_name in zip(activation_list, self.vae.activation_name_list):
                    results.update({'hist_activ_{}'.format(activation_name): activation})

                for weight, bias, activation_name in zip(weights, biases, self.vae.activation_name_list):
                    results.update({'hist_weights_{}'.format(activation_name): weight})
                    results.update({'hist_biasess_{}'.format(activation_name): bias})

                # save a few images
                # TODO(Since you're only extracting one image only feed one image!)
                img = self.vae.sess.run([self.vae.y], feed)
                # matplotlib.image.imsave(os.path.join(autoencoder_out, 'img_{}.png'.format(train_step)), img[0][0])
                results.update({'img_{}'.format(train_step): img[0][0]})

            return results

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/vae.json"
        self.vae.save_json(file_path)
        return file_path

    def _restore(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.vae
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            if not args.config:
                raise ValueError(
                    "Could not find params.pkl in either the checkpoint dir or "
                    "its parent directory.")
        else:
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        vae = ConvVAE(z_size=config['z_size'],
                      batch_size=config['batch_size'],
                      learning_rate=config['learning_rate'],
                      kl_tolerance=config['kl_tolerance'],
                      is_training=False,
                      reuse=False,
                      gpu_mode=False)  # use GPU on batchsize of 1000 -> much faster

        vae.load_json(os.path.join(path, 'vae.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=500, help='How long each rollout should be')
    parser.add_argument('--total_step_num', type=int, default=5000,
                        help='How many total steps to collect for the autoencoder training')
    parser.add_argument('--output_folder', type=str, default='~/sim2real')
    parser.add_argument('--gather_images', default=False, action='store_true',
                        help='Whether to gather images or just train')
    args = parser.parse_args()
    config = setup_sampling_env(parser)

    config={'z_size': 32, 'batch_size': 100, 'learning_rate': .0001, 'kl_tolerance': 0.5,
            'use_gpu': False, 'output_folder': args.output_folder}

    results = run(
        ConvVaeTrainer,
        name="autoencoder",
        stop={"training_iteration": 100},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        loggers=[TFLoggerPlus,] + list(DEFAULT_LOGGERS[0:2]),
        num_samples=1,
        config=config)
