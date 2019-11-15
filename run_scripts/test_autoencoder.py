import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from run_scripts.train_autoencoder import ConvVaeTrainer

def save_image(input, output, file_path):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(input[0])
    ax1.set_title('original')
    ax2.imshow(output[0])
    ax2.set_title('reconstruction')
    plt.savefig(file_path)
    plt.close(fig)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--output_folder', type=str, default='~/sim2real')
    args = parser.parse_args()

    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    config['gather_images'] = False
    config['run_mode'] = 'test'
    trainer = ConvVaeTrainer(config)
    trainer._restore(args.checkpoint)

    filepath = os.path.expanduser(config['output_folder'])
    perturb_path = os.path.join(os.path.expanduser((filepath)), 'autoencoder_perturb')

    if not os.path.exists(os.path.expanduser(perturb_path)):
        os.makedirs(perturb_path)

    # Lets do a few rounds of perturbing and save
    num_data = len(trainer.dataset)
    latent_size = config['z_size']
    for i in range(2):
        rand_int = np.random.randint(num_data)
        input = trainer.dataset[rand_int][np.newaxis, :]
        inner_layer = trainer.vae.encode(input)
        for j in range(latent_size + 1):
            if j == 0:
                perturb = 0.0 * np.eye(1, latent_size, k=j - 1)
            else:
                perturb = np.eye(1, latent_size, k=j - 1)
            inner_layer += perturb
            output = trainer.vae.decode(inner_layer)
            save_image(input, output, os.path.join(perturb_path, '{}_{}.jpg'.format(i, j)))


