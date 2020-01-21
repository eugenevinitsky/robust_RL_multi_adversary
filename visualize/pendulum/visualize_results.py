"""Point to a folder of results; it will loop through all the inner folders and generate appropriate bar graphs
   comparing the results"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_path', type=str, help='Pass the path to the folder containing all yuor results files')
parser.add_argument('--top_percent', type=float, default=1.0, help='The fraction of the top plots to plot')
parser.add_argument('--folder_names', nargs='+',
                    help='If passed, we only plot results if the hyperparam folder matches this')

parser.add_argument('--plot_names', nargs='+',
                    help='If passed, these are the names for the folder matching')


args = parser.parse_args()
prefix_list = ['base', 'adversary', 'friction_0p025', 'friction_0p05', 'friction_0p1', 'friction_0p2', 'friction_0p5',
               'friction_1p0', 'friction_2p0', 'friction_3p0', 'friction_5p0', 'friction_10p0',
               'gaussian_action_noise', 'gaussian_state_noise']
rew_results = [[] for _ in range(len(prefix_list))]

# Now lets walk through the folder
# TODO(@evinitsky) remove all these loops
for (dirpath, dirnames, filenames) in os.walk(args.results_path):
    for file in filenames:
        for i, prefix in enumerate(prefix_list):
            if prefix in file and '.png' not in file and file.split('.txt')[-2][-3:] == 'rew':
                file_output = np.loadtxt(os.path.join(dirpath, file))
                mean = np.mean(file_output)
                var = np.var(file_output)
                outer_folder = dirpath.split('/')[-1]
                rew_results[i].append((mean, var, file, outer_folder))


output_path = args.results_path.split('/')[-1]
if not os.path.exists('transfer_results/pendulum/{}'.format(output_path)):
    os.makedirs('transfer_results/pendulum/{}'.format(output_path))

#go through the results and pull out the one with the highest mean for a given experiment
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
    plt.figure(figsize=(40, 5))
    legends = []
    means = []
    vars = []
    outerfolders = []
    for mean, var, legend, outerfolder in result:
        means.append(mean)
        vars.append(var)
        legends.append(legend.split('.')[0].split('_' + prefix_list[i])[0])
        outerfolders.append(outerfolder)

    print('the winner for test {} with score {} is {} from folder {}'.format(prefix_list[i],
                                                                             np.max(means),
                                                                             legends[np.argmax(means)],
                                                                             outerfolders[np.argmax(means)]))

    # Now we sort
    means = np.array(means)
    vars = np.array(vars)
    legends = np.array(legends)
    inds = np.argsort(means)
    means = means[inds]
    vars = vars[inds]
    legends = legends[inds]
    means = means[-int(len(means) * args.top_percent):]
    vars = vars[-int(len(vars) * args.top_percent):]
    legends = legends[-int(len(legends) * args.top_percent):]
    y_pos = np.arange(len(legends))

    # plt.bar(y_pos, means, align='center', alpha=0.5, yerr=np.sqrt(vars))
    # Sometimes they don't fit
    bars_per_subplot = 6
    if len(means) > bars_per_subplot:
        # TODO(plot the leftovers)
        num_splits = len(means) // bars_per_subplot
        # handle the fact that there might be more than 6
        remainder = len(means) % bars_per_subplot
        split_len = int(len(means) / num_splits)
        fig, axs = plt.subplots(num_splits + (remainder > 0), 1)
        fig.set_figheight((5 + (remainder > 0)) * num_splits)
        fig.set_figwidth(50)
        for j in range(num_splits + (remainder > 0)):
            axs[j].bar(y_pos[j * split_len: split_len * (j + 1)], means[j * split_len: split_len * (j + 1)], align='center', alpha=0.5)
            axs[j].set_xticks(y_pos[j * split_len: split_len * (j + 1)])
            axs[j].set_xticklabels(legends[j * split_len: split_len * (j + 1)])
            axs[j].set_ylim(np.min(means), 0)

    else:
        plt.bar(y_pos, means, align='center', alpha=0.5)
        plt.xticks(y_pos, legends)
        plt.ylabel('Avg. score')
        plt.title('Score under {} test, top score is {}'.format(prefix_list[i], legends[np.argmax(means)]))

    plt.savefig('transfer_results/pendulum/{}/{}{}'.format(output_path, prefix_list[i], 'rew'))

# Now go through and plot all of them by folder name if so desired
if args.folder_names:

    if args.plot_names:
        legends = args.plot_names
    else:
        legends = args.folder_names

    new_results = [[0.0] * len(prefix_list) for i in range(len(legends))]
    for i, results in enumerate(rew_results):
        for result in results:
            if np.any([result[3] == folder_name for folder_name in args.folder_names]):
                exp_name = result[3]
                curr_index = 0
                for j, folder_name in enumerate(args.folder_names):
                    if folder_name == exp_name:
                        curr_index = j
                new_results[curr_index][i] = result[0]

    x = np.arange(len(prefix_list))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    fig.set_figwidth(25)

    width_range = np.linspace(-width * (len(legends) - 1) / 2 , width * (len(legends) - 1) / 2, len(legends))
    for trial_num, result in enumerate(new_results):
        if args.plot_names:
            rects1 = ax.bar(x + width_range[trial_num], result, width, label=args.plot_names[trial_num])
        else:
            rects1 = ax.bar(x + width_range[trial_num], result, width, label=''.join(args.folder_names[trial_num].split('_')[0:2]))

    ax.set_ylabel('Scores')
    ax.set_title('Blah')
    ax.set_xticks(x)
    # This is a hyper hand tuned way of labelling for the particular file names above
    xlabels = []
    for prefix in prefix_list:
        split = prefix.split('_')
        if len(split) == 1:
            xlabels.append(prefix)
        elif len(split) > 1 and 'noise' not in prefix:
            xlabels.append(''.join(prefix.split('_')[1:]))
        else:
            xlabels.append(prefix.split('_')[1][0]+ '_n')
    ax.set_xticklabels(xlabels)
    ax.legend()
    plt.savefig('transfer_results/pendulum/{}/{}'.format(output_path, 'folder_comparison'))
