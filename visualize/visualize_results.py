"""Point to a folder of results; it will loop through all the inner folders and generate appropriate bar graphs
   comparing the results"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')
parser.add_argument('--num_humans', type=str, default='1', help='Number of humans in experiment')


args = parser.parse_args()
prefix_list = ['base', 'friction', 'color', 'restrict_goal_region', 'chase_robot',
               'add_gaussian_noise_action', 'add_gaussian_noise_state', 'gaussian_state_action_noise']
rew_results = [[] for _ in range(len(prefix_list))]
steps_results = [[] for _ in range(len(prefix_list))]

base_value = 9
# Now lets walk through the folder
# TODO(@evinitsky) remove all these loops
for (dirpath, dirnames, filenames) in os.walk(args.results_path):
    for file in filenames:
        for i, prefix in enumerate(prefix_list):
            if prefix in file and '.png' not in file and file.split('.txt')[-2][-3:] == 'rew' and 'h'+args.num_humans in file:
                file_output = np.loadtxt(os.path.join(dirpath, file))
                mean = np.mean(file_output - base_value)
                var = np.var(file_output - base_value)
                rew_results[i].append((mean, var, file))
            elif prefix in file and '.png' not in file and file.split('.txt')[-2][-3:] != 'rew' and 'h'+args.num_humans in file:
                file_output = np.loadtxt(os.path.join(dirpath, file))
                mean = np.mean(file_output - base_value)
                var = np.var(file_output - base_value)
                steps_results[i].append((mean, var, file))

output_path = args.results_path.split('/')[-1]
if not os.path.exists('transfer_results/{}'.format(output_path)):
    os.makedirs('transfer_results/{}'.format(output_path))

# TODO(@evinitsky) go through the results and pull out the one with the highest mean for a given experiment
unique_rew_results = []
for result in rew_results:
    temp_results = []
    result_arr = np.array(result)
    if len(result_arr) > 0:
        names_arr = result_arr[:, 2]
        for name in np.unique(names_arr):
            indices = np.where(names_arr == name)
            max_value = np.argmax(np.squeeze(result_arr[indices, 0].astype(np.float)))
            temp_results.append(result[indices[0][max_value]])
        unique_rew_results.append(temp_results)

unique_step_results = []
for result in steps_results:
    temp_results = []
    result_arr = np.array(result)
    if len(result_arr) > 0:
        names_arr = result_arr[:, 2]
        for name in np.unique(names_arr):
            indices = np.where(names_arr == name)
            max_value = np.argmin(np.squeeze(result_arr[indices, 0].astype(np.float)))
            temp_results.append(result[indices[0][max_value]])
        unique_step_results.append(temp_results)

for i, result in enumerate(unique_rew_results):
    result = sorted(result, key=lambda x: x[2])
    plt.figure(figsize=(25, 5))
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

    plt.savefig('transfer_results/{}/{}{}'.format(output_path, prefix_list[i], 'rew'))

for i, result in enumerate(unique_step_results):
    result = sorted(result, key=lambda x: x[2])
    plt.figure(figsize=(25, 5))
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
    plt.ylabel('Avg. steps to goal')
    plt.title('Steps to goal under under {} test'.format(prefix_list[i]))

    plt.savefig('transfer_results/{}/{}{}'.format(output_path, prefix_list[i], 'steps'))