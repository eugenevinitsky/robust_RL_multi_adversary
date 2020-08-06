"""Point to folder with multiple mean, std across mass exp to make graph.
"""

from visualize.mujoco.transfer_tests import make_bernoulli_bandit_transfer_list
from utils.parsers import init_parser
import argparse
import os
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
font = {'family': 'normal',
        'size': 12}
matplotlib.rc('font', **font)

exps_to_plot = ['clustered_even', 'almost_zero_one']
exp_titles = ["Evenly Spread", "One good arm"]

arm10_vals = [np.linspace(0.0, 1.0, 10), np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])]
arm50_vals = [np.linspace(0.0, 1.0, 50), np.array([0.1]*49 + [0.9])]


def load_data(results_path):
    all_file_names = {}
    for (dirpath, dirnames, filenames) in os.walk(results_path):
        for run in filenames:
            if "sweep_rew.txt" in run and 'with_adv' not in run:
                tag = dirpath
                run_results = np.load(os.path.join(dirpath, run))
                all_file_names[tag] = run_results, dirpath

    return all_file_names


def load_total_transfer_scores(exp_path, num_arms=None, psr_only=False):
    exp_data = load_data(exp_path)

    parent_folders = []
    exp_total_scores = {}
    for file_name in exp_data:
        run_results, dirpath = exp_data[file_name]
        parent_folder = file_name.split('/')[-2]
        hyperparam_dir = file_name.split('/')[-1]
        if hyperparam_dir == 'rawscore':
            continue
        parent_folders.append(parent_folder)

        if num_arms and not psr_only:
            transfer_names = ['avg']
            means = [np.mean(run_results[:, 0])]
            transfer_names.extend([x[0] for x in make_bernoulli_bandit_transfer_list(num_arms)])
            means.extend(run_results[:, 0])
            assert len(transfer_names) == len(means)
        else:
            transfer_names = ['pseudorandom']
            means = [run_results[0, 0]]

        for name, mean in zip(transfer_names, means):
            if name in exp_total_scores:
                added = False
                for curr_parent_folder, mean_list, hyperparam_dirs in exp_total_scores[name]:
                    if curr_parent_folder == parent_folder:
                        mean_list.append(mean)
                        hyperparam_dirs.append(hyperparam_dir)
                        added = True
                if not added:
                    exp_total_scores[name].append((parent_folder, [mean], [hyperparam_dir]))
            else:
                exp_total_scores[name] = [(parent_folder, [mean], [hyperparam_dir])]

    return exp_total_scores


def autolabel(ax, rects, vals):
    """
    Attach a text label above each bar displaying its height
    """
    for rect, val in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width()/2., val,
                '%0.2f' % val,
                ha='center', va='bottom')


def save_barchart(total_scores, output_path, output_file_name, num_arms, separate_plots=False, group_seeds='mean', show=False, dist=7, yerr='std', psr_only=False):
    if separate_plots:
        for name, scores in total_scores.items():
            save_barchart({name: scores}, output_path, "{}/{}".format(output_file_name, name),
                          separate_plots=False, group_seeds=group_seeds, show=show)
    else:
        exps = total_scores.keys()
        if psr_only:
            fig, ax = plt.subplots(figsize=(15, 10))
        else:
            fig, ax = plt.subplots(figsize=(25, 10))
        vals_out = []
        labels_out = []
        exp_idx = 0
        for exp in exps:
            if exp not in exps_to_plot:
                continue
            results = total_scores[exp]
            if group_seeds == 'mean':
                labels = [result[0] for result in results]
                vals = [np.mean(result[1]) for result in results]
            elif group_seeds == 'max':
                labels = [result[0] for result in results]
                vals = [np.max(result[1]) for result in results]

            print(len(labels))
            print(labels)

            labels = ['_'.join(label.split("_")[9:-2]) if label != 'expert' else label for label in labels]
            labelidxs = np.argsort(labels)
            labels = [labels[idx] for idx in labelidxs]
            vals = [vals[idx] for idx in labelidxs]

            if yerr == 'std':
                err = [np.std(result[1])/np.sqrt(6) for result in results]
            elif yerr == 'min_max':
                err = np.array([[np.min(result[1]) - np.mean(result[1]), np.max(result[1]) -
                                 np.mean(result[1])] for result in results]).T
            elif yerr == 'off':
                err = 0

            colors = cm.rainbow(np.linspace(0, 1, len(labels)))
            if err is not 0:
                ax = plt.bar(0.5 + np.arange(len(labels)) + dist * exp_idx,
                             vals, color=colors, capsize=8, yerr=np.abs(err))
            else:
                ax = plt.bar(0.5 + np.arange(len(labels)) + dist * exp_idx, vals, color=colors)

            # autolabel(ax, ax, vals)
            vals_out.append(vals)
            labels_out.append(labels)
            exp_idx += 1

        plt.title('{}-armed Bandit Transfer Performance'.format(num_arms), fontsize=40, pad=25)
        if psr_only:
            plt.tick_params(labelbottom=False, bottom=False)
        else:
            plt.xticks(np.arange(len(exps_to_plot)) * dist + dist / 2, exp_titles, fontsize=35)
            # plt.xticks(np.arange(len(exps)) * dist + dist / 2, ["Avg.\nPerformance"] + ["Transfer\nTest {}".format(string.ascii_uppercase[i]) for i in range(len(exps) - 1)], fontsize=35)
        plt.yticks(fontsize=30)
        plt.ylabel("Mean score across seeds", fontsize=30)
        plt.legend(ax, ["RARL", "4 Adversaries", "10 Adversaries", "Domain Randomization", "Expert UCB1"], fontsize=25)
        plt.tight_layout()
        with open('{}/{}_transfer_performance.png'.format(output_path, output_file_name), 'wb') as png_out:
            plt.savefig(png_out)
        if show:
            plt.show()
        plt.close(fig)

    if num_arms == 10:
        arm_vals_list = arm10_vals
    elif num_arms == 50:
        arm_vals_list = arm50_vals

    for exp_name, arm_vals in zip(exp_titles, arm_vals_list):
        plt.title(exp_name, fontsize=40, pad=25)
        plt.bar(np.arange(len(arm_vals)), arm_vals)
        plt.tick_params(labelbottom=False, bottom=False)
        exp_idx += 1

        plt.tight_layout()
        plt.show()
        with open('{}/{}_arm_distribution.png'.format(output_path, exp_name), 'wb') as png_out:
            plt.savefig(png_out)
        plt.close(fig)


if __name__ == "__main__":
    parser = init_parser()
    parser.add_argument('experiment_path', type=str, default='transfer_out',
                        help='Path to experiment folder')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='')
    parser.add_argument('--num_arms', type=int,  required=False, help='Pull runlist to generate plot')
    parser.add_argument('--show_plots', action="store_true", help='Show plots as they are created.')
    parser.add_argument('--group_seeds', type=str,
                        choices=['mean', 'max', 'all'], default='mean', required=False, help='How to group seeds.')
    parser.add_argument('--yerr', choices=['std', 'min_max', 'off'], default='off',
                        required=False, help='How to represent the min_max error.')
    parser.add_argument('--spacing', type=int, default=6, help='Sets spacing between experiments')
    parser.add_argument('--psr_only', action='store_true', help='Sets spacing between experiments')
    args = parser.parse_args()
    total_exp_scores = load_total_transfer_scores(args.experiment_path, num_arms=args.num_arms, psr_only=args.psr_only)
    save_barchart(total_exp_scores, args.experiment_path, args.output_file_name,  num_arms=args.num_arms,
                  group_seeds=args.group_seeds, show=args.show_plots, yerr=args.yerr, dist=args.spacing, psr_only=args.psr_only)
