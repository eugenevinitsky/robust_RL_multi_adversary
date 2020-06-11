import os
import string

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from visualize.plot_heatmap import make_heatmap
from visualize.plot_heatmap import load_data, load_data_by_name
from visualize.mujoco.transfer_tests import cheetah_grid, cheetah_mass_sweep, ant_run_list, hopper_run_list, hopper_friction_sweep, hopper_mass_sweep, ant_mass_sweep, ant_friction_sweep


def generate_bar_plots(file_list, title, file_name, x_title=None, y_title=None, open_cmd=lambda x: np.load(x),
                       legend_rule=None, loc=None, y_lim=[], plot_std=False, fontsize=14, title_fontsize=16):
    plt.figure()
    if x_title:
        plt.xlabel(x_title)
    plt.ylabel(y_title, fontsize=fontsize)
    plt.title(title, fontsize=title_fontsize, pad=10)
    mean_list = []
    std_list = []
    ax_list = []
    spacing = 5.0
    colors = cm.rainbow(np.linspace(0, 1, len(file_list)))
    for i, file in enumerate(file_list):
        data = open_cmd(file)
        mean_list.append(np.mean(data))
        std_list.append(np.std(data))
        # ax = plt.bar(np.arange(3) + i * spacing, [np.mean(data), np.min(data), np.max(data)],  color=colors[i])
        # ax_list.append(ax)
    if plot_std:
        ax = plt.bar(np.arange(len(file_list)), mean_list, yerr=std_list, color=colors, capsize=3)
    else:
        ax = plt.bar(np.arange(len(file_list)), mean_list, color=colors)

    if loc:
        location_val = loc
    else:
        location_val = 0
    plt.legend(ax, legend_titles, loc=location_val, fontsize=fontsize)
    plt.xticks([], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if len(y_lim) > 0:
        plt.ylim(y_lim)
    # plt.xticks(np.arange(len(file_list)), tick_titles)
    # plt.tick_params(bottom=False)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_across_folders(folder_list, test_names, file_names, legend_names, open_cmd=lambda x: np.loadtxt(x),
                        fontsize=14, title_fontsize=16):
    test_results = np.zeros((len(test_names), len(folder_list)))
    colors = cm.rainbow(np.linspace(0, 1, len(folder_list)))
    for i, folder in enumerate(folder_list):
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for file in filenames:
                found_idx = [test_name in file and '.png' not in file for test_name in test_names]
                if np.sum(found_idx) > 0:
                    idx = np.where(np.array(found_idx) > 0)
                    test_idx = idx[0][0]
                    test_results[test_idx, i] = np.mean(open_cmd(os.path.join(dirpath, file)))

    for i, test in enumerate(test_names):
        plt.figure()
        ax = plt.bar(np.arange(len(folder_list)), test_results[i, :], color=colors, capsize=3)
        plt.tight_layout()
        plt.legend(ax, legend_names)
        plt.savefig(file_names[i])
        plt.close()

    # Now generate a plot for all of them
    dist = 7
    plt.figure()
    for i, test in enumerate(test_names):
        ax = plt.bar(np.arange(len(folder_list)) + dist * i, test_results[i, :], color=colors, capsize=3)
        plt.tight_layout()
        plt.legend(ax, legend_names)
    plt.xticks(fontsize=fontsize)
    plt.savefig(file_names[-1])
    plt.close()


def plot_across_seeds(outer_folder_list, test_names, file_names, legend_names, num_seeds, ylabel='Avg. Reward',
                      open_cmd=lambda x: np.loadtxt(x), yaxis=None, titles=[], fontsize=14, title_fontsize=16,
                      use_std=False, avg_across_tests=False, validation_set=False):
    test_results = np.zeros((len(test_names), len(outer_folder_list)))
    # indexed by [test_name, result_for given experiment]. Each internal element will be a list of length
    # number of seeds or number of hyperparameters
    std_deviations = [[[] for _ in range(len(test_names))]  for _ in range(len(outer_folder_list))]
    if validation_set:
        colors = cm.rainbow(np.linspace(0.1, 0.5, len(outer_folder_list)))
    else:
        colors = cm.rainbow(np.linspace(0.6, 1.0, len(outer_folder_list)))
    if not validation_set:
        use_std = False
    for i, folder in enumerate(outer_folder_list):
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for file in filenames:
                found_idx = ['_' + test_name + '_' in file and '.png' not in file for test_name in test_names]
                if np.sum(found_idx) > 0:
                    idx = np.where(np.array(found_idx) > 0)
                    test_idx = idx[0][0]
                    test_results[test_idx, i] += np.mean(open_cmd(os.path.join(dirpath, file)))
                    std_deviations[i][test_idx].append(np.mean(open_cmd(os.path.join(dirpath, file))))

    if avg_across_tests:
        std_deviations = np.sqrt(np.var(test_results, axis=0)/(num_seeds * len(test_names)))
        test_results = np.mean(test_results, axis=0) / num_seeds

        plt.figure()
        if use_std:
            # we compute the std deviation of the means across the seeds
            stds = std_deviations
            ax = plt.bar(np.arange(len(outer_folder_list)), test_results, color=colors,
                             capsize=3,
                             yerr=stds, alpha=0.7)
        else:
            ax = plt.bar(np.arange(len(outer_folder_list)), test_results, color=colors,
                             capsize=3, alpha=0.7)
        plt.grid(zorder=0, linestyle='-.', alpha=0.5)
        plt.tight_layout()
        plt.xticks(np.arange(len(legend_names)), legend_names, fontsize=fontsize,rotation=45)
        plt.ylabel(ylabel, fontsize=fontsize)
        if yaxis:
            plt.ylim(yaxis)
        plt.title(titles[-1], pad=10, fontsize=title_fontsize)
        plt.tight_layout()

        plt.savefig(file_names, bbox_inches='tight')
        plt.close()

        # Now write the means and standard deviations to a file for easy readout
        for i in range(len(outer_folder_list)):
            with open(file_names + '_' + legend_names[i] + '_' + 'mean_std.txt', 'w') as file:
                file.write('name mean std')
                file.write(str(test_results[i]) + ' ' + str(std_deviations[i]) + '\n')

    else:
        for i, test in enumerate(test_names):
            plt.figure()
            if use_std:
                ax = plt.bar(np.arange(len(outer_folder_list)), test_results[i, :] / num_seeds, color=colors, capsize=3, alpha=0.7)
            else:
                ax = plt.bar(np.arange(len(outer_folder_list)), test_results[i, :] / num_seeds, color=colors, capsize=3, alpha=0.7)
            plt.grid(zorder=0, linestyle='-.', alpha=0.5)
            plt.tight_layout()
            plt.legend(ax, legend_names, fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.tight_layout()
            plt.title(titles[i], fontsize=title_fontsize, pad=10)
            plt.savefig(file_names[i])

        # Now generate a plot for all of them
        dist = 7
        plt.figure()
        for i, test in enumerate(test_names):
            if use_std:
                # we compute the std deviation of the means across the seeds
                stds = [np.std(exp[i]) for exp in std_deviations]
                ax = plt.bar(np.arange(len(outer_folder_list)) + dist * i, test_results[i, :] / num_seeds, color=colors, capsize=3,
                             yerr=stds, alpha=0.7)
            else:
                ax = plt.bar(np.arange(len(outer_folder_list)) + dist * i, test_results[i, :] / num_seeds, color=colors, capsize=3, alpha=0.7)
            plt.grid(zorder=0, linestyle='-.', alpha=0.5)
            plt.tight_layout()
            plt.legend(ax, legend_names, fontsize=fontsize)
        plt.xticks([int(len(outer_folder_list) / 2) - 0.5 + (dist) * i for i in range(len(test_names))],
                   string.ascii_uppercase[0:len(test_names)], fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        if yaxis:
            plt.ylim(yaxis)
        plt.title(titles[-1], pad=10, fontsize=title_fontsize)
        plt.tight_layout()

        plt.savefig(file_names[-1], bbox_inches='tight')
        plt.close()

        # Now write the means and standard deviations to a file for easy readout
        for i in range(len(outer_folder_list)):
            with open(file_names[-1] + '_' + legend_names[i] + '_' + 'mean_std.txt', 'w') as file:
                file.write('name mean std')
                for j, test_name in enumerate(test_names):
                    file.write(test_name + ' ' + str(np.mean(std_deviations[i][j])) + ' ' + str(np.std(std_deviations[i][j])) + '\n')


def plot_across_seeds_heatmap(exp_type, mass_sweep, friction_sweep, outer_folder_list, test_names, file_name, num_seeds,
                      titles=[], open_cmd=lambda x: np.loadtxt(x), fontsize=22, title_fontsize=16):

    test_results = np.zeros((len(test_names), len(outer_folder_list)))
    for i, folder in enumerate(outer_folder_list):
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for file in filenames:
                found_idx = ['_' + test_name + '_' in file and '.png' not in file for test_name in test_names]
                if np.sum(found_idx) > 0:
                    idx = np.where(np.array(found_idx) > 0)
                    test_idx = idx[0][0]
                    test_results[test_idx, i] += np.mean(open_cmd(os.path.join(dirpath, file)))
    for i in range(len(outer_folder_list)):
        means = test_results[:,i].reshape(len(mass_sweep), len(friction_sweep)) / num_seeds
        fig = plt.figure()
        if exp_type == 'hopper':
            plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
            plt.title(titles[i], fontsize=title_fontsize)
            plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x) for x in mass_sweep], fontsize=10)
            plt.ylabel("Mass coef", fontsize=fontsize)
            plt.xticks(ticks=np.arange(len(friction_sweep))[0::2], labels=["{:0.2f}".format(x) for x in friction_sweep][0::2], fontsize=10)
            plt.xlabel("Friction coef", fontsize=fontsize)
        elif exp_type == 'cheetah':
            plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=2000, vmax=7000)
            plt.title(titles[i], fontsize=title_fontsize)
            plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x) for x in mass_sweep], fontsize=10)
            plt.ylabel("Mass coef", fontsize=fontsize)
            plt.xticks(ticks=np.arange(len(friction_sweep)), labels=["{:0.2f}".format(x) for x in friction_sweep], fontsize=10)
            plt.xlabel("Friction coef", fontsize=fontsize)
        elif exp_type == 'ant':
            plt.imshow(means.T, interpolation='nearest', cmap='seismic', aspect='equal', vmin=2000, vmax=7000)
            plt.title(titles[i], fontsize=title_fontsize)
            plt.yticks(ticks=np.arange(len(mass_sweep)), labels=["{:0.2f}".format(x) for x in mass_sweep], fontsize=10)
            plt.ylabel("Mass coef", fontsize=fontsize)
            plt.xticks(ticks=np.arange(len(friction_sweep)), labels=["{:0.2f}".format(x) for x in friction_sweep], fontsize=10)
            plt.xlabel("Friction coef", fontsize=fontsize)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(file_name[i], bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    ### CONSTANTS
    fontsize = 22
    title_fontsize = 24
    ###################################################################################################################
    ######################################### CHEETAH #########################################################
    ###################################################################################################################

    ###########################################################
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/cheetah')) + '/'

    ##############################################################
    # generate the bar charts comparing 0 adv, dr, and 5 adv for hopper
    # generate validation set bar charts
    test_names = []
    cheetah_friction_sweep_good = np.linspace(0.5, 1.5, 11)
    cheetah_friction_sweep_bad = np.linspace(0.1, 0.9, 11)
    transfer_test_names_good = []
    transfer_test_names_bad = []

    cheetah_grid_good = np.meshgrid(cheetah_mass_sweep, cheetah_friction_sweep_good)
    for mass, fric in np.vstack((cheetah_grid_good[0].ravel(), cheetah_grid_good[1].ravel())).T:
        transfer_test_names_good.append('m_{}_f_{}'.format(mass, fric))

    cheetah_grid_bad = np.meshgrid(cheetah_mass_sweep, cheetah_friction_sweep_bad)
    for mass, fric in np.vstack((cheetah_grid_bad[0].ravel(), cheetah_grid_bad[1].ravel())).T:
        transfer_test_names_bad.append('m_{}_f_{}'.format(mass, fric))

    output_files = 'final_plots/cheetah/hc_compare_valid_all_seeds_good.png'
    file_names_good = [data_dir + 'hc_0adv_concat1_seed_good/',
                  data_dir + 'hc_0adv_concat1_seed_dr_good/',
                  data_dir + 'hc_1adv_concat1_seed_str0p1_good/',
                  data_dir + 'hc_3adv_concat1_seed_str0p1_norew_good/',
                  data_dir + 'hc_5adv_concat1_seed_str0p1_norew_good/']
    legend_names = ['0 Adv', 'DR', '1 Adv', '3 Adv', '5 Adv']
    titles = ['Cheetah, Valid. Set Reward - Good Param.']

    plot_across_seeds(file_names_good, transfer_test_names_good, output_files, legend_names, num_seeds=10, yaxis=[0, 7000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, validation_set=True, use_std=True)
    output_files = ['final_plots/cheetah/hc_heat0adv_good.png', 'final_plots/cheetah/hc_heatdr_good.png', 'final_plots/cheetah/hc_heat1adv_good.png', 'final_plots/cheetah/hc_heat3adv_good.png', 'final_plots/cheetah/hc_heat5adv_good.png']
    plot_across_seeds_heatmap(exp_type='cheetah', mass_sweep=cheetah_mass_sweep, friction_sweep=cheetah_friction_sweep_good,  outer_folder_list=file_names_good, test_names=transfer_test_names_good, file_name=output_files,
                       titles=legend_names, num_seeds=10, title_fontsize=title_fontsize)


    output_files = 'final_plots/cheetah/hc_compare_valid_all_seeds_bad.png'
    file_names_bad = [data_dir + 'hc_0adv_concat1_seed_bad/',
                  data_dir + 'hc_0adv_concat1_seed_dr_bad/',
                  data_dir + 'hc_1adv_concat1_seed_str0p1_bad/',
                  data_dir + 'hc_3adv_concat1_seed_str0p1_norew_bad/',
                  data_dir + 'hc_5adv_concat1_seed_str0p1_norew_bad/']
    legend_names = ['0 Adv', 'DR', '1 Adv', '3 Adv', '5 Adv']
    titles = ['Cheetah, Valid. Set Reward - Bad Param.']

    plot_across_seeds(file_names_bad, transfer_test_names_bad, output_files, legend_names, num_seeds=10, yaxis=[0, 7000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, validation_set=True, use_std=True)
    output_files = ['final_plots/cheetah/hc_heat0adv_bad.png', 'final_plots/cheetah/hc_heatdr_bad.png', 'final_plots/cheetah/hc_heat1adv_bad.png', 'final_plots/cheetah/hc_heat3adv_bad.png', 'final_plots/cheetah/hc_heat5adv_bad.png']
    plot_across_seeds_heatmap(exp_type='cheetah', mass_sweep=cheetah_mass_sweep, friction_sweep=cheetah_friction_sweep_bad,  outer_folder_list=file_names_bad, test_names=transfer_test_names_bad, file_name=output_files,
                      titles=legend_names, num_seeds=10, title_fontsize=title_fontsize)

    # generate the test set maps for the best validation set result
    test_names = [
        'friction_hard_torsoheadfthighmax',
        'friction_hard_floorheadfshinmax',
        'friction_hard_bthighbshinbfootmax',
        'friction_hard_floortorsoheadmax',
        'friction_hard_floorbshinffootmax',
        'friction_hard_bthighbfootffootmax',
        'friction_hard_bthighfthighfshinmax',
        'friction_hard_headfshinffootmax'
    ]

    #good
    output_files = 'final_plots/cheetah/hc_avg_test_all_seed_good.png'
    titles = ['Cheetah, Test Set Reward - Good Param.']
    plot_across_seeds(file_names_good, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 7000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize,avg_across_tests=True, use_std=True)

    output_files = ['final_plots/cheetah/hc_seed_good_' + name for name in test_names]
    output_files.append('final_plots/cheetah/compare_test_all_seed_good')
    titles = ['blah' for i in range(len(test_names))] + ['Average reward on test set across 10 seeds']

    plot_across_seeds(file_names_good, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 8000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize)
    output_files[-1] = 'final_plots/cheetah/compare_test_all_seed_std_good'
    plot_across_seeds(file_names_good, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 8000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize,use_std=True)

    #bad
    output_files = 'final_plots/cheetah/hc_avg_test_all_seed_bad.png'
    titles = ['Cheetah, Test Set Reward - Bad Param.']
    plot_across_seeds(file_names_bad, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 7000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, use_std=True)

    output_files = ['final_plots/cheetah/hc_seed_bad_' + name for name in test_names]
    output_files.append('final_plots/cheetah/compare_test_all_seed_bad')
    titles = ['blah' for i in range(len(test_names))] + ['Average reward on test set across 10 seeds']

    plot_across_seeds(file_names_bad, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 8000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize)
    output_files[-1] = 'final_plots/cheetah/compare_test_all_seed_std_bad'
    plot_across_seeds(file_names_bad, test_names, output_files, legend_names, num_seeds=10,yaxis=[0, 8000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, use_std=True)

    ###################################################################################################################
    ######################################### ANT #########################################################
    ###################################################################################################################

    ###########################################################
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/ant')) + '/'

    ##############################################################
    # generate the bar charts comparing 0 adv, dr, and 5 adv for hopper
    # generate validation set bar charts
    transfer_test_names = []

    ant_grid = np.meshgrid(ant_mass_sweep, ant_friction_sweep)
    for mass, fric in np.vstack((ant_grid[0].ravel(), ant_grid[1].ravel())).T:
        transfer_test_names.append('m_{}_f_{}'.format(mass, fric))


    output_files = 'final_plots/ant/ant_compare_valid_all_seeds.png'
    file_names = [data_dir + 'ant_0adv_concat1_seed/',
                       data_dir + 'ant_0adv_concat1_seed_dr/',
                       data_dir + 'ant_1adv_concat1_seed_str0p15/',
                       data_dir + 'ant_3adv_concat1_seed_str0p15_norew/',
                       data_dir + 'ant_5adv_concat1_seed_str0p15_norew/']
    legend_names = ['0 Adv', 'DR', '1 Adv', '3 Adv', '5 Adv']
    titles = ['Ant, Validation Set Reward']

    plot_across_seeds(file_names, transfer_test_names, output_files, legend_names, num_seeds=10,
                      yaxis=[0, 7000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, validation_set=True, use_std=True)
    output_files = ['final_plots/ant/ant_heat0adv.png', 'final_plots/ant/ant_heatdr.png', 'final_plots/ant/ant_heat1adv.png', 'final_plots/ant/ant_heat3adv.png', 'final_plots/ant/ant_heat5adv.png']

    plot_across_seeds_heatmap(exp_type='ant', mass_sweep=ant_mass_sweep, friction_sweep=ant_friction_sweep,  outer_folder_list=file_names, test_names=transfer_test_names, file_name=output_files,
                      titles=legend_names, num_seeds=10, title_fontsize=title_fontsize)

    # generate the test set maps for the best validation set result
    test_names = [
        'friction_hard_flla1a3max',
        'friction_hard_torsoa1rblmax',
        'friction_hard_frla2blmax',
        'friction_hard_torsoflla1max',
        'friction_hard_flla2a4max',
        'friction_hard_frlbla4max',
        'friction_hard_frla3rblmax',
        'friction_hard_a1rbla4max'
    ]

    output_files = 'final_plots/ant/ant_avg_test_all_seed.png'
    titles = ['Ant, Test Set Reward']
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 7000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, use_std=True)

    output_files = ['final_plots/ant/ant_seed_' + name for name in test_names]
    output_files.append('final_plots/ant/compare_test_all_seed')
    titles = ['blah' for i in range(len(test_names))] + ['Average reward on test set across 10 seeds']

    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 8000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize)
    output_files[-1] = 'final_plots/ant/compare_test_all_seed_std'
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 8000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, use_std=True)

    ###################################################################################################################
    ######################################### HOPPER #########################################################
    ###################################################################################################################

    ###########################################################
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/hopper')) + '/'

    ##############################################################
    # generate the bar charts comparing 0 adv, dr, and 5 adv for hopper
    # generate validation set bar charts
    transfer_test_names = []

    hop_grid = np.meshgrid(hopper_mass_sweep, hopper_friction_sweep)
    for mass, fric in np.vstack((hop_grid[0].ravel(), hop_grid[1].ravel())).T:
        transfer_test_names.append('m_{}_f_{}'.format(mass, fric))

    output_files = 'final_plots/hopper/hop_compare_valid_all_seeds.png'
    file_names = [data_dir + 'hop_0adv_concat1_seed/',
                       data_dir + 'hop_0adv_concat1_seed_dr/',
                       data_dir + 'hop_1adv_concat1_seed_str0p25/',
                       data_dir + 'hop_3adv_concat1_seed_str0p25/',
                       data_dir + 'hop_5adv_concat1_seed_str0p25_norew/']
    legend_names = ['0 Adv', 'DR', '1 Adv', '3 Adv', '5 Adv']
    titles = ['Hopper, Validation Set Reward']

    plot_across_seeds(file_names, transfer_test_names, output_files, legend_names, num_seeds=10,
                      yaxis=[0, 3000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, validation_set=True, use_std=True)

    # generate the test set maps for the best validation set result
    test_names = [
        'friction_hard_torsolegmax_floorthighfootmin',
        'friction_hard_floorthighmax_torsolegfootmin',
        'friction_hard_footlegmax_floortorsothighmin',
        'friction_hard_torsothighfloormax_footlegmin',
        'friction_hard_torsofootmax_floorthighlegmin',
        'friction_hard_floorthighlegmax_torsofootmin',
        'friction_hard_floorfootmax_torsothighlegmin',
        'friction_hard_thighlegmax_floortorsofootmin',
    ]

    output_files = 'final_plots/hopper/hop_avg_test_all_seed.png'
    titles = ['Hopper, Test Set Reward']
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, use_std=True)
    output_files = ['final_plots/hopper/hop_heat0adv.png', 'final_plots/hopper/hop_heatdr.png', 'final_plots/hopper/hop_heat1adv.png', 'final_plots/hopper/hop_heat3adv.png', 'final_plots/hopper/hop_heat5adv.png']
    plot_across_seeds_heatmap(exp_type='hopper', mass_sweep=hopper_mass_sweep, friction_sweep=hopper_friction_sweep,  outer_folder_list=file_names, test_names=transfer_test_names, file_name=output_files,
                      titles=legend_names, num_seeds=10, title_fontsize=title_fontsize)

    output_files = ['final_plots/hopper/hop_seed_' + name for name in test_names]
    output_files.append('final_plots/hopper/compare_test_all_seed')
    titles = ['blah' for i in range(len(test_names))] + ['Average reward on test set across 10 seeds']

    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize)
    output_files[-1] = 'final_plots/hopper/compare_test_all_seed_std'
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, use_std=True)

    ###################################################################################################################
    ######################################### HOPPER NUMADV #########################################################
    ###################################################################################################################

    ###########################################################
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/hopper')) + '/'

    ##############################################################
    # generate the bar charts comparing 0 adv, dr, and 5 adv for hopper
    # generate validation set bar charts
    transfer_test_names = []
    for test in hopper_run_list:
        transfer_test_names.append(test[0])

    output_files = 'final_plots/hopper/hop_compare_valid_all_seeds_numadv.png'
    file_names = [data_dir + 'hop_0adv_concat1_seed/',
                  data_dir + 'hop_1adv_concat1_seed_str0p25/',
                  data_dir + 'hop_2adv_concat1_seed_str0p25/',
                  data_dir + 'hop_3adv_concat1_seed_str0p25/',
                  data_dir + 'hop_5adv_concat1_seed_str0p25_norew/',
                  data_dir + 'hop_7adv_concat1_seed_str0p25_norew/',
                  data_dir + 'hop_9adv_concat1_seed_str0p25_norew/',
                  data_dir + 'hop_11adv_concat1_seed_str0p25_norew/']
    legend_names = ['0 Adv', '1 Adv', '2 Adv', '3 Adv', '5 Adv', '7 Adv', '9 Adv', '11 Adv']
    titles = ['Hopper, Validation Set Reward']

    plot_across_seeds(file_names, transfer_test_names, output_files, legend_names, num_seeds=10,
                      yaxis=[0, 3000], titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, validation_set=True, use_std=True)

    # generate the test set maps for the best validation set result
    test_names = [
        'friction_hard_torsolegmax_floorthighfootmin',
        'friction_hard_floorthighmax_torsolegfootmin',
        'friction_hard_footlegmax_floortorsothighmin',
        'friction_hard_torsothighfloormax_footlegmin',
        'friction_hard_torsofootmax_floorthighlegmin',
        'friction_hard_floorthighlegmax_torsofootmin',
        'friction_hard_floorfootmax_torsothighlegmin',
        'friction_hard_thighlegmax_floortorsofootmin',
    ]

    # good
    output_files = 'final_plots/hopper/hop_avg_test_all_seed_numadv.png'
    titles = ['Hopper, Test Set Reward']
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, avg_across_tests=True, use_std=True)

    output_files = ['final_plots/hopper/hop_test_seed_numadv_' + name for name in test_names]
    output_files.append('final_plots/hopper/compare_test_all_seed_numadv')
    titles = ['blah' for i in range(len(test_names))] + ['Average reward on test set across 10 seeds']

    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize)
    output_files[-1] = 'final_plots/hopper/compare_test_all_seed_std_numadv'
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3000],
                      titles=titles,
                      fontsize=fontsize, title_fontsize=title_fontsize, use_std=True)

    ###################################################################################################################
    ######################################### HOPPER POLICY CORRELATION MATRIX #########################################
    ###################################################################################################################
    adv_1_file = data_dir + 'policy_correlation_data/results_1adv.txt'

    with open(adv_1_file, 'rb') as file:
        adv_1_results = np.loadtxt(file)

    plt.figure()
    plt.tight_layout()
    plt.imshow(adv_1_results, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
    plt.colorbar()
    fontsize = 18
    title_fontsize = 20
    plt.yticks(ticks=np.arange(adv_1_results.shape[1]))
    plt.xticks(ticks=np.arange(adv_1_results.shape[0]))
    plt.ylabel('agent seed index', fontsize=fontsize)
    plt.xlabel('adversary seed index', fontsize=fontsize)
    plt.title('1 Adversary', fontsize=title_fontsize)
    plt.savefig('final_results/final_plots/hopper/adv_1_correlation.png', bbox_inches="tight")

    adv_3_file = data_dir + 'policy_correlation_data/results_3adv.txt'

    with open(adv_3_file, 'rb') as file:
        adv_3_results = np.loadtxt(file)

    plt.figure()
    plt.imshow(adv_3_results, interpolation='nearest', cmap='seismic', aspect='equal', vmin=400, vmax=3600)
    plt.tight_layout()
    plt.colorbar()
    fontsize = 18
    title_fontsize = 20
    plt.yticks(ticks=np.arange(adv_3_results.shape[1]))
    plt.xticks(ticks=np.arange(adv_3_results.shape[0]))
    plt.ylabel('agent seed index', fontsize=fontsize)
    plt.xlabel('adversary seed index', fontsize=fontsize)
    plt.title('3 Adversaries', fontsize=title_fontsize)
    plt.savefig('final_results/final_plots/hopper/adv_3_correlation.png', bbox_inches="tight")
