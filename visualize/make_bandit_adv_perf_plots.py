"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
import os
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
font = {'family' : 'normal',
        'size' : 12}
matplotlib.rc('font', **font)
from utils.parsers import init_parser
from visualize.plot_heatmap import load_data
from visualize.pendulum.transfer_tests import make_bandit_transfer_list


parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('adv_mean_sweep_location', type=str, help='Location of adv_mean_sweep file.')
parser.add_argument('output_file', help='output png file loc')
args = parser.parse_args()

with open(args.adv_mean_sweep_location, 'rb') as file:
        adv_mean_sweep = np.loadtxt(file)

means = np.array(adv_mean_sweep)[:,0]
num_adversaries = len(means)
adv_names = ['adversary{}'.format(adv_num) for adv_num in range(num_adversaries)]
fig = plt.figure()
plt.bar(np.arange(num_adversaries), means)
plt.title("Regret playing against each adversary.")
plt.xticks(np.arange(num_adversaries), adv_names)
plt.xlabel("Adversary")
plt.ylabel("Avg regret")
with open(args.adv_mean_sweep_location, 'wb') as file:
        plt.savefig(file)
plt.close(fig)
