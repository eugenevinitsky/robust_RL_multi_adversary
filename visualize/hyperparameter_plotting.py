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
    for file_name in exp_data:
        means, _ = exp_data[file_name]
        if base_exp:
            means = means - base_data[max_base_data][0]
        total_transfer_score = np.mean(means)
        exp_total_scores[file_name] = total_transfer_score
    
    save_barchart(exp_total_scores, output_path, exp_path, exp_name, show)

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
        fig = plt.figure(figsize=(18,4))
        bar_plot = plt.bar(range(len(total_scores)), list(total_scores.values()), yerr=np.std(list(total_scores.values())),align='center')
        plt.xticks(range(len(total_scores)), list([key[:6] for key in total_scores.keys()]))
        plt.xlabel("Hyperparameter run")
        plt.ylabel("Mean score across swept transfer values")
        ax = plt.gca()
        autolabel(ax, bar_plot, list(total_scores.values()))
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


