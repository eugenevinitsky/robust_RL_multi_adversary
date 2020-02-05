"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.parsers import init_parser
from visualize.plot_heatmap import load_data
from visualize.pendulum.transfer_tests import make_bandit_transfer_list

def plot_total_transfer_scores(output_path, output_file_name, exp_path, group_seeds, num_arms=None, base_exp=None, show=False):
    exp_data = load_data(exp_path)

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
                    if group_seeds and curr_parent_folder == parent_folder:
                        mean_list.append(mean)
                        hyperparam_dirs.append(hyperparam_dir)
                        added = True
                if not added:
                    exp_total_scores[name].append((parent_folder, [mean], [hyperparam_dir]))
            else:
                exp_total_scores[name] = [(parent_folder, [mean], [hyperparam_dir])]
    
    if group_seeds:
        names = parent_folders
    else:
        names = exp_data.keys()

    save_barchart(exp_total_scores, names, output_path, output_file_name, show)

def save_barchart(total_scores, file_names, output_path, output_file_name, show):
    for name, scores in total_scores.items():
        with open('{}/{}{}_{}.png'.format(output_path, output_file_name, name, "transfer_heatmap"),'wb') as heatmap:
            fig, ax = plt.subplots(figsize=(18,10))
            names = [score[0] for score in scores]
            means = [np.mean(score[1]) for score in scores]
            best_hyperparams = [(np.argmax(score[1]), np.max(score[1]), score[2][np.argmax(score[1])]) for score in scores]
            for parent_folder, (idx, score, hyperparam_dir) in zip(names, best_hyperparams):
                print(parent_folder, hyperparam_dir, score)
            ax.set_ylim([np.min([np.min(score[1]) for score in scores]) * 1.2, 0])
            err = np.array([[np.min(score[1]) - np.mean(score[1]), np.max(score[1])- np.mean(score[1])] for score in scores]).T
            ax.bar(np.arange(len(means)), means, yerr=np.abs(err), capsize=10)
            plt.title('Comparing performance on {} transfer test'.format(name))
            # plt.xticks(np.arange(len(means)), [name[10:20] for name in file_names])
            plt.xticks(np.arange(len(means)), [name for name in names], rotation=70)
            plt.xlabel("Transfer run for {}".format(name))
            plt.ylabel("Mean score across swept")
            ax = plt.gca()
            plt.tight_layout()
            plt.legend()
            plt.savefig(heatmap)
            if show:
                plt.show()
            plt.close(fig)

if __name__ == "__main__":
    output_path = os.path.expanduser('~/transfer_results/')

    parser = init_parser()
    parser.add_argument('experiment_path', type=str, default='transfer_out',
                        help='Path to experiment folder')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='')
    parser.add_argument('--num_arms', type=int,  required=False, help='Pull runlist to generate plot')
    parser.add_argument('--show_plots', action="store_true", help='Show plots as they are created.')
    parser.add_argument('--group_seeds', action="store_true", help='Show plots as they are created.')
    args = parser.parse_args()
    plot_total_transfer_scores(args.experiment_path, args.output_file_name, args.experiment_path, args.group_seeds, num_arms=args.num_arms,  show=args.show_plots)


