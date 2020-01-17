"""Point to folder with multiple mean, std across mass exp to make graph.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')


args = parser.parse_args()

x_axis = np.linspace(0.1, 15.0, 15)
print(len(x_axis))
all_runs = {}

for (dirpath, dirnames, filenames) in os.walk(args.results_path):
    for run in filenames:
        if "sweep" in run:
            run_results = np.load(os.path.join(dirpath, run))

            means = run_results[1:, 0]
            stds = run_results[1:, 1]

            all_runs[run] = (means, stds)


plt.figure()
for run in all_runs:
    import ipdb; ipdb.set_trace()
    means, stds = all_runs[run]
    plt.plot(x_axis, means, label=run)
    plt.fill_between(x_axis, means + stds, means - stds, alpha=0.7)

plt.legend()
plt.show()
            
