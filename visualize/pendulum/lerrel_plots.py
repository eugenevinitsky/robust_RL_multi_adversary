"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')
parser.add_argument('output_path', type=str, help='Pass the path to the folder containing all yuor results files')


args = parser.parse_args()

mass_sweep = np.linspace(.7, 1.3, 11)
friction_sweep = np.linspace(0.5, 1.5, 11)
# print(len(x_axis))
all_runs = {}

for (dirpath, dirnames, filenames) in os.walk(args.results_path):
    for run in filenames:
        if "sweep" in run:
            tag = dirpath.split("/")[-1][:5]
            run_results = np.load(os.path.join(dirpath, run))

            means = run_results[1:, 0]
            stds = run_results[1:, 1]

            all_runs[tag] = (means, stds)


# plt.figure()
# for run in all_runs:
#     import ipdb; ipdb.set_trace()
#     means, stds = all_runs[run]
#     plt.plot(x_axis, means, label=run)
    # plt.fill_between(x_axis, means + stds, means - stds, alpha=0.7)

for run in all_runs:
    # base = np.array(temp_output)[0,0]
    means, _ = all_runs[run]
    means = means.reshape(len(mass_sweep), len(friction_sweep))
    with open('{}/{}_{}.png'.format(args.output_path, run.split("sweep")[0], "transfer_robustness"),'wb') as transfer_robustness:
        fig = plt.figure()
        plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
        plt.title(run)
        plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x * 3.53) for x in mass_sweep])
        plt.ylabel("Mass coef")
        plt.xticks(ticks=np.arange(len(friction_sweep)), labels=["{:0.2f}".format(x) for x in friction_sweep])
        plt.xlabel("Friction coef")
        plt.colorbar()
        plt.savefig(transfer_robustness)
        # plt.show()

# plt.legend()

            
