"""Point to a folder of results; it will loop through all the inner folders and generate appropriate bar graphs
   comparing the results"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')

args = parser.parse_args()
prefix_list = ['base', 'friction', 'color', 'unrestrict_goal_reg', 'chase_robot',
               'gaussian_state_noise', 'gaussian_action_noise', 'gaussian_state_action_noise']
results = [[] for _ in range(len(prefix_list))]

base_value = 9
# Now lets walk through the folder
# TODO(@evinitsky) remove all these loops
for (dirpath, dirnames, filenames) in os.walk(args.results_path):
    for file in filenames:
        for i, prefix in enumerate(prefix_list):
            if prefix in file and '.png' not in file:
                file_output = np.loadtxt(os.path.join(dirpath, file))
                mean = np.mean(file_output - base_value)
                var = np.var(file_output - base_value)
                results[i].append((mean, var, file))

output_path = args.results_path.split('/')[-1]
if not os.path.exists('transfer_results/{}'.format(output_path)):
    os.makedirs('transfer_results/{}'.format(output_path))

for i, result in enumerate(results):
    result = sorted(result, key=lambda x: x[2])
    plt.figure(figsize=(20, 5))
    legends = []
    means = []
    vars = []
    for mean, var, legend in result:
        means.append(mean)
        vars.append(var)
        legends.append(legend.split('.')[0].split('_' + prefix_list[i])[0])
    y_pos = np.arange(len(legends))

    # plt.bar(y_pos, means, align='center', alpha=0.5, yerr=np.sqrt(vars))
    plt.bar(y_pos, means, align='center', alpha=0.5)
    plt.xticks(y_pos, legends)
    plt.ylabel('Avg. score')
    plt.title('Score under {} test'.format(prefix_list[i]))

    plt.savefig('transfer_results/{}/{}'.format(output_path, prefix_list[i]))
