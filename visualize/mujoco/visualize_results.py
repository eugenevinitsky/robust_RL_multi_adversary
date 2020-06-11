"""Point to a folder of results; it will loop through all the inner folders and generate appropriate bar graphs
   comparing the results"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')


args = parser.parse_args()
prefix_list = ['base', 'friction', 'gaussian_action_noise', 'gaussian_state_noise']
rew_results = [[] for _ in range(len(prefix_list))]

# Now lets walk through the folder
for (dirpath, dirnames, filenames) in os.walk(args.results_path):
    for file in filenames:
        for i, prefix in enumerate(prefix_list):
            if prefix in file and '.png' not in file and file.split('.txt')[-2][-3:] == 'rew':
                file_output = np.loadtxt(os.path.join(dirpath, file))
                mean = np.mean(file_output)
                var = np.var(file_output)
                rew_results[i].append((mean, var, file))


output_path = args.results_path.split('/')[-1]
if not os.path.exists('transfer_results/pendulum/{}'.format(output_path)):
    os.makedirs('transfer_results/pendulum/{}'.format(output_path))

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

    plt.savefig('transfer_results/pendulum/{}/{}{}'.format(output_path, prefix_list[i], 'rew'))