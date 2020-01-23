"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO(@evinitsky) PUT THIS BACK DUDE PLEASE REMEMBER
hopper_mass_sweep = np.linspace(.7, 1.3, 11)
hopper_friction_sweep = np.linspace(0.7, 1.3, 11)

def load_data(results_path):
    all_file_names = {}
    for (dirpath, dirnames, filenames) in os.walk(results_path):
        for run in filenames:
            if "sweep" in run:
                tag = dirpath.split("/")[-1]
                run_results = np.load(os.path.join(dirpath, run))

                means = run_results[1:, 0]
                stds = run_results[1:, 1]

                all_file_names[tag] = (means, stds)

    return all_file_names

def make_heatmap(results_path, output_path, show=False):
    sweep_data = load_data(results_path)
    for file_name in sweep_data:
        print(file_name)
        means, _ = sweep_data[file_name]
        means = means.reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))
        save_heatmap(means, hopper_mass_sweep, hopper_friction_sweep, output_path, file_name.split("sweep")[0], show)

def save_heatmap(means, mass_sweep, friction_sweep, output_path, file_name, show):
    with open('{}/{}_{}.png'.format(output_path, file_name, "transfer_heatmap"),'wb') as heatmap:
        fig = plt.figure()
        plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
        plt.title(file_name)
        plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x * 3.53) for x in mass_sweep])
        plt.ylabel("Mass coef")
        plt.xticks(ticks=np.arange(len(friction_sweep)), labels=["{:0.2f}".format(x) for x in friction_sweep])
        plt.xlabel("Friction coef")
        plt.colorbar()
        plt.savefig(heatmap)
        if show:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')
    parser.add_argument('output_path', type=str, help='Output file location.')
    parser.add_argument('--show_images', action="store_true", help='Show plots as they are created.')
    args = parser.parse_args()

    make_heatmap(args.results_path, args.output_path, args.show_images)


