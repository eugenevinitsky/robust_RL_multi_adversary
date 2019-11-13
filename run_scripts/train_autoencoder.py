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
                                    ob[:, :, 0:3] + (128 / 255))
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
        raw_data = plt.imread(os.path.join(filepath, filename), format='jpg')[:, :, 0:3]
        data[idx] = raw_data
        idx += 1
        if ((i + 1) % 100 == 0):
            print("loading file", i + 1)
    return data


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


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

        if config['gather_images']:
            gather_images(config['env_config'], images_path, config['horizon'], config['total_step_num'])

        # Now lets train the auto-encoder
        np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

        # Hyperparameters for ConvVAE
        self.batch_size = config['batch_size']

        # Parameters for training

        # # TODO(@evinitsky) maybe don't load the whole dataset into memory wtf
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
                           gpu_mode=config['use_gpu'],
                           top_percent=config['top_percent'])

    def _train(self):
        # train loop:
        np.random.shuffle(self.dataset)
        for idx in range(self.num_batches):
            batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
            obs = batch.astype(np.float) / 255.0

            feed = {self.vae.x: obs, }

            # (train_loss, r_loss, kl_loss, train_step, grads_and_vars, activation_list, _) = self.vae.sess.run([
            #     self.vae.loss, self.vae.r_loss, self.vae.kl_loss, self.vae.global_step, self.vae.grads,
            #     self.vae.activation_list, self.vae.train_op
            # ], feed)

            (train_loss, r_loss, train_step, grads_and_vars, activation_list, _) = self.vae.sess.run([
                self.vae.loss, self.vae.r_loss, self.vae.global_step, self.vae.grads,
                self.vae.activation_list, self.vae.train_op
            ], feed)

            # TODO(@evinitsky) add gradient norm, add histogram of activations and weights

            results = {
                "epoch": self.iteration,
                "train_loss": train_loss,
                "r_loss": r_loss,
                # "kl_loss": kl_loss,
            }

            grad_squares = [np.square(g).sum() for (g, v) in grads_and_vars]
            grad_weights = grad_squares[::2]
            grad_biases = grad_squares[1::2]
            for weight, bias, activation_name in zip(grad_weights, grad_biases, self.vae.activation_name_list):
                results.update({"layer_{}_grad_norm".format(activation_name): np.sqrt(weight)})
                results.update({"bias_{}_grad_norm".format((activation_name)): np.sqrt(bias)})

            vars = [v for (g, v) in grads_and_vars]
            weights = vars[::2]
            biases = vars[1::2]

            if ((train_step + 1) % train_config['img_freq'] == 0):
                # Save histograms of the activations
                # Okay, now lets make a hist of the activations and a hist of the weights
                for activation, activation_name in zip(activation_list, self.vae.activation_name_list):
                    results.update({'hist_activ_{}'.format(activation_name): activation})

                for weight, bias, activation_name in zip(weights, biases, self.vae.activation_name_list):
                    results.update({'hist_weights_{}'.format(activation_name): weight})
                    results.update({'hist_biases_{}'.format(activation_name): bias})

                # save a few images
                # TODO(Since you're only extracting one image only feed one image!)
                img = self.vae.sess.run([self.vae.y], feed)
                # matplotlib.image.imsave(os.path.join(autoencoder_out, 'img_{}.png'.format(train_step)), img[0][0])

                # construct a matplotlib image with the two side by side

                fig = plt.figure(figsize=(8, 16))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(img[0][0])
                ax1.set_title('reconstruction')
                ax2.imshow(obs[0])
                ax2.set_title('original')
                data = fig2data(fig)
                plt.close(fig)

                results.update({'img_{}'.format(train_step): data})

            return results

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/vae.json"
        self.vae.save_json(file_path)
        return file_path

    def _restore(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.vae
        config_dir = os.path.dirname(path)
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
        self.vae = ConvVAE(z_size=config['z_size'],
                      batch_size=config['batch_size'],
                      learning_rate=config['learning_rate'],
                      kl_tolerance=config['kl_tolerance'],
                      is_training=False,
                      reuse=False,
                      gpu_mode=False)  # use GPU on batchsize of 1000 -> much faster

        self.vae.load_json(os.path.join(path, 'vae.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=500, help='How long each rollout should be')
    parser.add_argument('--total_step_num', type=int, default=200,
                        help='How many total steps to collect for the autoencoder training')
    parser.add_argument('--output_folder', type=str, default='~/sim2real')
    parser.add_argument('--gather_images', default=False, action='store_true',
                        help='Whether to gather images or just train')
    parser.add_argument('--img_freq', type=int, default=40,
                        help='How often to log the autoencoder image output')
    parser.add_argument('--top_percent', type=float, default=0.1,
                        help='Which percent of the loss tensor to keep in the loss. For example, 0.1 will'
                             'keep only the top 10% largest elements of the loss')
    args = parser.parse_args()
    env_config = setup_sampling_env(parser)

    train_config = {'z_size': 12, 'batch_size': 100, 'learning_rate': .0001 * (1 / args.top_percent), 'kl_tolerance': 0.5,
                    'use_gpu': False, 'output_folder': args.output_folder, 'img_freq': args.img_freq,
                    'env_config': env_config, 'top_percent': args.top_percent, 'gather_images': args.gather_images,
                    'horizon': args.horizon, 'total_step_num': args.total_step_num}

    results = run(
        ConvVaeTrainer,
        name="autoencoder",
        stop={"training_iteration": 1000},
        checkpoint_freq=args.img_freq,
        checkpoint_at_end=True,
        loggers=[TFLoggerPlus, ] + list(DEFAULT_LOGGERS[0:2]),
        num_samples=1,
        config=train_config)
