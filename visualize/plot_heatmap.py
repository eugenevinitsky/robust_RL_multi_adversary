"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
from collections import OrderedDict
import os

import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size' : 16}
matplotlib.rc('font', **font)

import numpy as np

hopper_mass_sweep = np.linspace(.7, 1.3, 11)
hopper_friction_sweep = np.linspace(0.7, 1.3, 11)

cheetah_mass_sweep = np.linspace(.5, 1.5, 11)
cheetah_friction_sweep_good = np.linspace(0.5, 1.5, 11)
cheetah_friction_sweep = np.linspace(0.1, 0.9, 11)

ant_mass_sweep = np.linspace(.5, 1.5, 11)
ant_friction_sweep = np.linspace(0.1, 0.9, 11)

def load_data(results_path):
    all_file_names = OrderedDict()
    for (dirpath, dirnames, filenames) in os.walk(results_path):
        for run in filenames:
            if "sweep_rew.txt" in run and not "adv_mean_sweep_rew" in run:
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

def load_bandit_data(results_path):
    all_file_names = OrderedDict()
    for (dirpath, dirnames, filenames) in os.walk(results_path):
        for run in filenames:
            if "sweep_rew.txt" in run and 'with_adv' not in run:
                tag = dirpath
                run_results = np.load(os.path.join(dirpath, run))
                all_file_names[tag] = run_results, dirpath
    return all_file_names
    

def load_data_by_name(results_path, name):
    """This is used for the test set"""
    all_file_names = OrderedDict()
    for (dirpath, dirnames, filenames) in os.walk(results_path):
        for run in filenames:
            if name in run and 'png' not in run:
                tag = dirpath.split("/")[-1]
                try:
                    run_results = np.load(os.path.join(dirpath, run))
                except:
                    run_results = np.loadtxt(os.path.join(dirpath, run))

                all_file_names[tag] = (np.mean(run_results), np.std(run_results))

    return all_file_names


def make_heatmap(results_path, exp_type, output_path, show=False, output_file_name=None, fontsize=14, title_fontsize=16):
    sweep_data = load_data(results_path)
    for file_name in sweep_data:
        print(file_name)
        _, _, _, _, means, _, _, _, dirpath = sweep_data[file_name]

        if not output_path:
            output_name = dirpath
        else:
            output_name = output_path

        if not output_file_name:
            output_file_name = file_name.split("sweep")[0]

        if exp_type == 'hopper':
            means = means.reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))
            save_heatmap(means, hopper_mass_sweep, hopper_friction_sweep, output_name,
                         output_file_name, show, exp_type, fontsize, title_fontsize)
        elif exp_type == 'cheetah':
            means = means.reshape(len(cheetah_mass_sweep), len(cheetah_friction_sweep))
            save_heatmap(means, cheetah_mass_sweep, cheetah_friction_sweep, output_name,
                         output_file_name, show, exp_type, fontsize, title_fontsize)
        elif exp_type == 'ant':
            means = means.reshape(len(ant_mass_sweep), len(ant_friction_sweep))
            save_heatmap(means, ant_mass_sweep, ant_friction_sweep, output_name,
                         output_file_name, show, exp_type, fontsize, title_fontsize)



def save_heatmap(means, mass_sweep, friction_sweep, output_path, file_name, show, exp_type, fontsize=14, title_fontsize=16):
    # with open('{}/{}_{}.png'.format(output_path, file_name, "transfer_heatmap"),'wb') as heatmap:
    fig = plt.figure()
    if exp_type == 'hopper':
        plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
        plt.title(file_name, fontsize=title_fontsize)
        plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x * 3.53) for x in mass_sweep])
        plt.ylabel("Mass coef", fontsize=fontsize)
        plt.xticks(ticks=np.arange(len(friction_sweep))[0::2], labels=["{:0.2f}".format(x) for x in friction_sweep][0::2])
        plt.xlabel("Friction coef", fontsize=fontsize)
    elif exp_type == 'cheetah':
        plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=2000, vmax=8000)
        plt.title(file_name, fontsize=title_fontsize)
        plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x * 6.0) for x in mass_sweep])
        plt.ylabel("Mass coef", fontsize=fontsize)
        plt.xticks(ticks=np.arange(len(friction_sweep)), labels=["{:0.2f}".format(x) for x in friction_sweep])
        plt.xlabel("Friction coef", fontsize=fontsize)
    elif exp_type == 'ant':
        plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=8000)
        plt.title(file_name, fontsize=title_fontsize)
        plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x * 6.0) for x in mass_sweep])
        plt.ylabel("Mass coef", fontsize=fontsize)
        plt.xticks(ticks=np.arange(len(friction_sweep)), labels=["{:0.2f}".format(x) for x in friction_sweep])
        plt.xlabel("Friction coef", fontsize=fontsize)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('{}/{}_{}.png'.format(output_path.replace(' ', '_'), file_name, "transfer_heatmap"))
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')
    parser.add_argument('exp_type', type=str, help='hopper, cheetah, pendulum, ant')
    parser.add_argument('--output_path', type=str, help='Output file location.')
    parser.add_argument('--show_images', action="store_true", help='Show plots as they are created.')
    args = parser.parse_args()

    make_heatmap(args.results_path, args.exp_type, args.output_path, args.show_images)


