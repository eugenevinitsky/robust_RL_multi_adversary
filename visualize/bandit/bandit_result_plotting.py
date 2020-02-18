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
from visualize.plot_heatmap import load_bandit_data
from visualize.pendulum.transfer_tests import make_bandit_transfer_list

def load_total_transfer_scores(exp_path, num_arms=None):
    exp_data = load_bandit_data(exp_path)

    parent_folders = []
    exp_total_scores = {}
    for file_name in exp_data:
        run_results, dirpath = exp_data[file_name]
        parent_folder = file_name.split('/')[-2]
        hyperparam_dir = file_name.split('/')[-1]
        parent_folders.append(parent_folder)

        transfer_names = ['avg']
        means = [np.mean(run_results[:,0])]

        if num_arms:
            transfer_names.extend([x[0] for x in make_bandit_transfer_list(num_arms)])
            means.extend(run_results[:,0])
            assert len(transfer_names) == len(means)
        else:
            transfer_names.append('pseudorandom')
            means.append(run_results[0,0])

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
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*val,
                '%d' % int(val),
                ha='center', va='bottom')

def save_barchart(total_scores, output_path, output_file_name, num_arms, separate_plots=False, group_seeds='mean', show=False, dist=7, yerr='std'):
    if separate_plots:
        for name, scores in total_scores.items():
            save_barchart({name: scores}, output_path, "{}/{}".format(output_file_name, name), separate_plots=False, group_seeds=group_seeds, show=show)
    else:
        exps = total_scores.keys()
        fig, ax = plt.subplots(figsize=(25,10))
        for exp_idx, exp in enumerate(exps):
            results = total_scores[exp]
            if group_seeds == 'mean':
                labels = [result[0] for result in results]
                vals = [np.mean(result[1]) for result in results]
            elif group_seeds == 'max':
                labels = [result[0] for result in results]
                vals = [np.max(result[1]) for result in results]
            
            print(len(labels))

            if yerr == 'std':
                err = [np.std(result[1]) for result in results]
            elif yerr == 'min_max':
                err = np.array([[np.min(result[1]) - np.mean(result[1]), np.max(result[1])- np.mean(result[1])] for result in results]).T
            elif yerr == 'off':
                err = 0
            
            colors = cm.rainbow(np.linspace(0, 1, len(labels)))
            if err:
                ax = plt.bar(0.5 + np.arange(len(labels)) + dist * exp_idx, vals, color=colors, capsize=3, yerr=np.abs(err))
            else:
                ax = plt.bar(0.5 + np.arange(len(labels)) + dist * exp_idx, vals, color=colors)

        plt.title('{}-armed Bandit Transfer Performance'.format(num_arms), fontsize=40, pad=25)
        plt.xticks(np.arange(len(exps)) * dist + dist / 2, ["Avg.\nPerformance"] + ["Transfer\nTest {}".format(string.ascii_uppercase[i]) for i in range(len(exps) - 1)], fontsize=35)
        plt.yticks(fontsize=30)
        plt.ylabel("Mean score across seeds", fontsize=30)
        plt.legend(ax, ["RARL", "4 Adversaries", "10 Adversaries", "20 Adversaries", "Domain Randomization"], fontsize=25)
        with open('{}/{}_transfer_performance.png'.format(output_path, output_file_name),'wb') as png_out:
            plt.savefig(png_out)
        if show:
            plt.show()
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
    parser.add_argument('--group_seeds', choices=['mean', 'max', 'all'], default='mean', required=False, help='How to group seeds.')
    parser.add_argument('--yerr', choices=['std', 'min_max', 'off'], default='off', required=False, help='How to represent the min_max error.')
    parser.add_argument('--spacing', type=int, default=6, help='Sets spacing between experiments')
    args = parser.parse_args()
    total_exp_scores = load_total_transfer_scores(args.experiment_path, num_arms=args.num_arms)
    save_barchart(total_exp_scores, args.experiment_path, args.output_file_name,  num_arms=args.num_arms, group_seeds=args.group_seeds, show=args.show_plots, yerr=args.yerr, dist=args.spacing)


