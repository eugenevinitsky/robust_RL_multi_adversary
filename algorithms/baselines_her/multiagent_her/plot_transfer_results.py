import argparse
from collections import OrderedDict
import os

import numpy as np
import matplotlib.pyplot as plt

def load_data(results_path):
    all_file_names = OrderedDict()
    for (dirpath, dirnames, filenames) in os.walk(results_path):
        for run in filenames:
            tag = dirpath.split("/")[-1]
            try:
                run_results = np.load(os.path.join(dirpath, run))
            except:
                run_results = np.loadtxt(os.path.join(dirpath, run))

            base_score = run_results[0, 0]
            base_std = run_results[0, 1]
            base_steps = run_results[0, 2]
            base_steps_std = run_results[0, 3]
            means = run_results[1:, 0]
            stds = run_results[1:, 1]
            step_means = run_results[1:, 2]
            step_stds = run_results[1:, 3]
            all_file_names[tag] = (base_score, base_std, base_steps, base_steps_std, means, stds, step_means, step_stds, dirpath)

    return all_file_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder', type=str)
    args = parser.parse_args()
    all_file_names = load_data(args)
    means = means.reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))

