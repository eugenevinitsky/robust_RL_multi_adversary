"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from visualize.plot_heatmap import load_data

def plot_total_transfer_scores(output_path, exp_name, exp_path, base_exp=None, show=False):
    exp_data = load_data(exp_path)
    max_base_data = None
    if base_exp:
        base_data = load_data(base_exp)
        base_keys = list(base_data.keys())
        max_base_data = max(base_keys, key=lambda x: np.mean(base_data[x][0]))


    exp_total_scores = {}
    exp_total_steps = {}
    for file_name in exp_data:
        base_score, base_std, base_steps, base_steps_std, means, mean_stds, step_means, step_stds, _ = exp_data[file_name]
        if base_exp:
            means = means - base_data[max_base_data][0]
        total_transfer_score = np.mean(means)
        exp_total_scores[file_name] = {'base_score': base_score,
                                       'base_std': base_std,
                                       'transfer_scores': total_transfer_score,
                                       'transfer_std': mean_stds}

        total_transfer_steps = np.mean(step_means)
        exp_total_steps[file_name] = {'base_score': base_steps, 'transfer_scores': total_transfer_steps,
                                      'base_std': base_steps_std, 'transfer_std': step_stds}

    save_barchart(exp_total_scores, output_path, exp_path, exp_name, show)

    save_barchart(exp_total_steps, output_path, exp_path, exp_name + 'steps', show)

def autolabel(ax, rects, vals):
    """
    Attach a text label above each bar displaying its height
    """
    for rect, val in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*val,
                '%d' % int(val),
                ha='center', va='bottom')

def save_barchart(total_scores, output_path, exp_path, file_name, show):
    if not output_path:
        output_path = exp_path
    with open('{}/{}_{}.png'.format(output_path, file_name, "transfer_heatmap"),'wb') as heatmap:
        fig, ax = plt.subplots(figsize=(18,4))
        total_scores = {key.split('Multi')[-1]: val for key, val in total_scores.items()}
        if isinstance(total_scores, dict):
            transfer_means = {key: val['transfer_scores'] for key, val in total_scores.items()}
            base_score = {key: val['base_score'] for key, val in total_scores.items()}
            base_std = {key: val['base_std'] for key, val in total_scores.items()}
            transfer_std = {key: val['transfer_std'] for key, val in total_scores.items()}

            width = 0.35
            ax.bar(np.arange(len(total_scores)) - width / 2, list(base_score.values()), width,
                   yerr=list(base_std.values()), label='base score', align='center')
            ax.bar(np.arange(len(total_scores)) + width / 2, list(transfer_means.values()), width,
                   label='transfer means', align='center')
            # ax.bar(np.arange(len(total_scores)) - width / 2, list(base_score.values()), width,
            #        label='base score', align='center')
            # ax.bar(np.arange(len(total_scores)) + width / 2, list(transfer_means.values()), width,
            #        label='transfer means', align='center')
            max_base = np.argmax(list(base_score.values()))
            plt.title('Base score vs. transfer mean, {}, top_score: {}, {}'.format(file_name,
                                                                             list(transfer_means.values())[max_base],
                                                                             list([key[:6] for key in total_scores.keys()])[max_base]))
        else:
            bar_plot = ax.bar(range(len(total_scores)), list(total_scores.values()), yerr=np.std(list(total_scores.values())),align='center')
        plt.xticks(range(len(total_scores)), list([key[:6] for key in total_scores.keys()]))
        plt.xlabel("Hyperparameter run")
        plt.ylabel("Mean score across swept transfer values")
        ax = plt.gca()
        plt.legend()
        # autolabel(ax, bar_plot, list(total_scores.values()))
        plt.savefig(heatmap)
        if show:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str, help='Pass the path to the folder containing all your results files')
    parser.add_argument('--output_path', type=str, help='Output file location.')
    parser.add_argument('--base_experiment', required=False, type=str, help='subtract this experiment\'s heatmap from the others before taking the mean\
                                                                            if folder, find best base')
    parser.add_argument('--show_plots', action="store_true", help='Show plots as they are created.')
    args = parser.parse_args()

    plot_total_transfer_scores(args.output_path, os.path.basename(args.experiment_path), args.experiment_path, base_exp=args.base_experiment, show=args.show_plots)


