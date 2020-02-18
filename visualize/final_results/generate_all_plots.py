import os
import string

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from visualize.linear_env.test_eigenvals import plot_eigenvals
from visualize.plot_heatmap import make_heatmap


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


def plot_across_seeds(outer_folder_list, test_names, file_names, legend_names, num_seeds, ylabel='Avg. Score',
                      open_cmd=lambda x: np.loadtxt(x), yaxis=None, titles=[], fontsize=14, title_fontsize=16,
                      use_std=False):
    test_results = np.zeros((len(test_names), len(outer_folder_list)))
    # indexed by [test_name, result_for given experiment]. Each internal element will be a list of length
    # number of seeds or number of hyperparameters
    std_deviations = [[[] for _ in range(len(test_names))]  for _ in range(len(outer_folder_list))]

    colors = cm.rainbow(np.linspace(0, 1, len(outer_folder_list)))
    for i, folder in enumerate(outer_folder_list):
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for file in filenames:
                found_idx = [test_name in file and '.png' not in file for test_name in test_names]
                if np.sum(found_idx) > 0:
                    idx = np.where(np.array(found_idx) > 0)
                    test_idx = idx[0][0]
                    test_results[test_idx, i] += np.mean(open_cmd(os.path.join(dirpath, file)))
                    std_deviations[i][test_idx].append(np.mean(open_cmd(os.path.join(dirpath, file))))


    for i, test in enumerate(test_names):
        plt.figure()
        if use_std:
            # TODO(@evinitsky) add error bars
            ax = plt.bar(np.arange(len(outer_folder_list)), test_results[i, :] / num_seeds, color=colors, capsize=3)
        else:
            ax = plt.bar(np.arange(len(outer_folder_list)), test_results[i, :] / num_seeds, color=colors, capsize=3)
        plt.tight_layout()
        plt.legend(ax, legend_names)
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
                         yerr=stds)
        else:
            ax = plt.bar(np.arange(len(outer_folder_list)) + dist * i, test_results[i, :] / num_seeds, color=colors, capsize=3)
        plt.tight_layout()
        plt.legend(ax, legend_names)
    plt.xticks([int(len(outer_folder_list) / 2) - 0.5 + (dist) * i for i in range(len(test_names))],
               string.ascii_uppercase[0:len(test_names)], fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if yaxis:
        plt.ylim(yaxis)
    plt.title(titles[-1], pad=10, fontsize=title_fontsize)
    plt.tight_layout()

    plt.savefig(file_names[-1])

    # Now write the means and standard deviations to a file for easy readout
    for i in range(len(outer_folder_list)):
        with open(file_names[-1] + '_' + legend_names[i] + '_' + 'mean_std.txt', 'w') as file:
            file.write('name mean std')
            for j, test_name in enumerate(test_names):
                file.write(test_name + ' ' + str(np.mean(std_deviations[i][j])) + ' ' + str(np.std(std_deviations[i][j])) + '\n')



if __name__ == '__main__':
    ### CONSTANTS
    fontsize = 14
    title_fontsize = 16

    ###################################################################################################################
    ######################################### LINEAR SYSTEMS #########################################################
    ###################################################################################################################
    # Generate eigenvalue plots for the linear env
    plot_eigenvals(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_plots/linear_env'))

    ################################################### 2 DIMENSIONS #################################################
    # Generate the bar plots for the linear system, 2d, base score
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/linear')) + '/'
    file_list = [data_dir + 'linear_1adv_d2_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-05_01-35-57j2kpm975/'
                            'linear_1adv_d2_conc100_h100_r1_base_sweep_rew',
                 data_dir + 'linear_5adv_d2_conc100_h100_low600_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-36-474brj302h/linear_5adv_d2_conc100_h100_low600_r1_base_sweep_rew',
                 data_dir + 'linear_10adv_d2_conc100_h100_low600_r2/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_06-07-15y_pz__fl/linear_10adv_d2_conc100_h100_low600_r2_base_sweep_rew',
                 data_dir + 'linear_20adv_d2_conc100_h100_low600_r2/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-05_06-30-289nslo5d6/linear_20adv_d2_conc100_h100_low600_r2_base_sweep_rew',
                 data_dir + 'linear_dr_d2_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-48-04sj3wxtih/linear_dr_d2_conc100_h100_r1_base_sweep_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['RARL', '5 Adversaries', '10 Adversaries', '20 Adversaries', 'Domain randomization']
    title = 'Regret for base system, Dim 2'
    file_name = 'final_plots/linear_env/base_regret_2d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title, loc=3, y_lim=[-800, 0], fontsize=fontsize)

    # Generate the bar plots for the linear system, 2d, base domain randomization score
    file_list = [data_dir + 'linear_1adv_d2_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-05_01-35-57j2kpm975/'
                            'linear_1adv_d2_conc100_h100_r1_domain_rand_rew',
                 data_dir + 'linear_5adv_d2_conc100_h100_low600_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-36-474brj302h/linear_5adv_d2_conc100_h100_low600_r1_domain_rand_rew',
                 data_dir + 'linear_10adv_d2_conc100_h100_low600_r2/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_06-07-15y_pz__fl/linear_10adv_d2_conc100_h100_low600_r2_domain_rand_rew',
                 data_dir + 'linear_20adv_d2_conc100_h100_low600_r2/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-05_06-30-289nslo5d6/linear_20adv_d2_conc100_h100_low600_r2_domain_rand_rew',
                 data_dir + 'linear_dr_d2_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-48-04sj3wxtih/linear_dr_d2_conc100_h100_r1_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['RARL', '5 Adversaries', '10 Adversaries', '20 Adversaries', 'Domain randomization']
    title = 'Regret for domain randomization, Dim 2'
    file_name = 'final_plots/linear_env/drand_regret_2d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title, loc=3, y_lim=[-500, 0], fontsize=fontsize)

    # Generate the bar plots for the linear system, 2d, hard domain randomization score
    file_list = [data_dir + 'linear_1adv_d2_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-05_01-35-57j2kpm975/'
                            'linear_1adv_d2_conc100_h100_r1_hard_domain_rand_rew',
                 data_dir + 'linear_5adv_d2_conc100_h100_low600_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-36-474brj302h/linear_5adv_d2_conc100_h100_low600_r1_hard_domain_rand_rew',
                 data_dir + 'linear_10adv_d2_conc100_h100_low600_r2/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_06-07-15y_pz__fl/linear_10adv_d2_conc100_h100_low600_r2_hard_domain_rand_rew',
                 data_dir + 'linear_20adv_d2_conc100_h100_low600_r2/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-05_06-30-289nslo5d6/linear_20adv_d2_conc100_h100_low600_r2_hard_domain_rand_rew',
                 data_dir + 'linear_dr_d2_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-48-04sj3wxtih/linear_dr_d2_conc100_h100_r1_hard_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['RARL', '5 Adversaries', '10 Adversaries', '20 Adversaries', 'Domain randomization']
    title = 'Regret for unstable systems, Dim 2'
    file_name = 'final_plots/linear_env/hard_regret_2d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title, loc=3, y_lim=[-1600, 0], fontsize=fontsize)

    ################################################### 4 DIMENSIONS #################################################

    # Generate the bar plots for the linear system, 4d, base score
    file_list = [data_dir + 'linear_1adv_d4_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-05_01-40-58sn2t_tzy/'
                            'linear_1adv_d4_conc100_h100_r1_base_sweep_rew',
                 data_dir + 'linear_5adv_d4_conc100_h100_low4=1000_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-40-25uy86jcys/linear_5adv_d4_conc100_h100_low4=1000_r1_base_sweep_rew',
                 data_dir + 'linear_10adv_d4_conc100_h100_low1000_r2/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_06-12-4429_l_j25/linear_10adv_d4_conc100_h100_low1000_r2_base_sweep_rew',
                 # data_dir + 'linear_20adv_d4_conc100_h100_low1000_r2/'
                 #            'PPO_0_lambda=0.5,lr=0.0005_2020-02-05_06-15-40s7htro4g/linear_20adv_d4_conc100_h100_low1000_r2_base_sweep_rew'
                 data_dir + 'linear_dr_d4_conc100_h100_r1/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-05_01-49-31s9wzxlln/linear_dr_d4_conc100_h100_r1_base_sweep_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['RARL', '5 Adversaries', '10 Adversaries', 'Domain randomization']
    title = 'Regret for base system, Dim 4'
    file_name = 'final_plots/linear_env/base_regret_4d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title, loc=3, y_lim=[-850, 0], fontsize=fontsize)

    # Generate the bar plots for the linear system, 4d, base domain randomization score
    file_list = [data_dir + 'linear_1adv_d4_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-05_01-40-58sn2t_tzy/'
                            'linear_1adv_d4_conc100_h100_r1_domain_rand_rew',
                 data_dir + 'linear_5adv_d4_conc100_h100_low4=1000_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-40-25uy86jcys/linear_5adv_d4_conc100_h100_low4=1000_r1_domain_rand_rew',
                 data_dir + 'linear_10adv_d4_conc100_h100_low1000_r2/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_06-12-4429_l_j25/linear_10adv_d4_conc100_h100_low1000_r2_domain_rand_rew',
                 # data_dir + 'linear_20adv_d4_conc100_h100_low1000_r2/'
                 #            'PPO_0_lambda=0.5,lr=0.0005_2020-02-05_06-15-40s7htro4g/linear_20adv_d4_conc100_h100_low1000_r2_domain_rand_rew',
                 data_dir + 'linear_dr_d4_conc100_h100_r1/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-05_01-49-31s9wzxlln/linear_dr_d4_conc100_h100_r1_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['RARL', '5 Adversaries', '10 Adversaries','Domain randomization']
    title = 'Regret for domain randomization, Dim 4'
    file_name = 'final_plots/linear_env/drand_regret_4d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title, loc=3, y_lim=[-700, 0], fontsize=fontsize)

    # Generate the bar plots for the linear system, 4d, hard domain randomization score
    file_list = [data_dir + 'linear_1adv_d4_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-05_01-40-58sn2t_tzy/'
                            'linear_1adv_d4_conc100_h100_r1_hard_domain_rand_rew',
                 data_dir + 'linear_5adv_d4_conc100_h100_low4=1000_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_01-40-25uy86jcys/linear_5adv_d4_conc100_h100_low4=1000_r1_hard_domain_rand_rew',
                 data_dir + 'linear_10adv_d4_conc100_h100_low1000_r2/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-05_06-12-4429_l_j25/linear_10adv_d4_conc100_h100_low1000_r2_hard_domain_rand_rew',
                 # data_dir + 'linear_20adv_d4_conc100_h100_low1000_r2/'
                 #            'PPO_0_lambda=0.5,lr=0.0005_2020-02-05_06-15-40s7htro4g/linear_20adv_d4_conc100_h100_low1000_r2_hard_domain_rand_rew',
                 data_dir + 'linear_dr_d4_conc100_h100_r1/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-05_01-49-31s9wzxlln/linear_dr_d4_conc100_h100_r1_hard_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['RARL', '5 Adversaries', '10 Adversaries', 'Domain randomization']
    title = 'Regret for unstable systems, Dim 4'
    file_name = 'final_plots/linear_env/hard_regret_4d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title, loc=3, y_lim=[-1750, 0], fontsize=fontsize)


    ###################################################################################################################
    ######################################### HOPPER #########################################################
    ###################################################################################################################

    ###########################################################
    fontsize = 16
    title_fontsize = 20
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/hopper')) + '/'

    # generate the relevant heatmaps that will go into the paper
    # 0 adversary
    make_heatmap(data_dir + 'hop_0adv_concat1_seed_lv0p9_lr0005/PPO_2_seed=2_2020-02-02_22-49-49107ecvph/', 'hopper',
                 'final_plots/hopper', output_file_name='0 Adversaries', fontsize=fontsize, title_fontsize=title_fontsize)

    # 1 adversary
    make_heatmap(data_dir + 'hop_1adv_concat1_seed_str0p25_lv0p0_lr0005/PPO_9_seed=9_2020-02-02_22-53-39u2gwqrg8/', 'hopper',
                 'final_plots/hopper', output_file_name='1 Adversary', fontsize=fontsize, title_fontsize=title_fontsize)

    # 5 adversary
    make_heatmap(data_dir + 'hop_5adv_concat10_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/PPO_0_seed=0_2020-02-02_22-56-06h0gae7mk', 'hopper',
                 'final_plots/hopper', output_file_name='5 Adversaries', fontsize=fontsize, title_fontsize=title_fontsize)

    # Domain randomization
    make_heatmap(data_dir + 'hop_0adv_concat10_seed_dr_lv0p5_lr00005/PPO_0_seed=0_2020-02-02_23-06-31qtb9bzqs', 'hopper',
                 'final_plots/hopper', output_file_name='Domain Randomization', fontsize=fontsize, title_fontsize=title_fontsize)

    ##############################################################
    # generate the bar charts comparing 0 adv, dr, and 5 adv for hopper

    # generate the test set maps for the best validation set result
    curr_path = os.path.abspath(__file__)
    data_dir = os.path.abspath(os.path.join(curr_path, '../../../data/hopper')) + '/'
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
    output_files = ['final_plots/hopper/hop' + name for name in test_names]
    output_files.append('final_plots/hopper/compare_all')

    file_names = [data_dir + 'hop_0adv_concat1_seed_lv0p9_lr0005/PPO_2_seed=2_2020-02-02_22-49-49107ecvph/',
                  data_dir + 'hop_1adv_concat1_seed_str0p25_lv0p0_lr0005/PPO_9_seed=9_2020-02-02_22-53-39u2gwqrg8',
                  data_dir + 'hop_0adv_concat10_seed_dr_lv0p5_lr00005/PPO_0_seed=0_2020-02-02_23-06-31qtb9bzqs',
                  data_dir + 'hop_5adv_concat10_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/PPO_0_seed=0_2020-02-02_22-56-06h0gae7mk']
    legend_names = ['0 Adversary', 'RARL No Memory', 'DR', '5 Adv Memory']
    plot_across_folders(file_names, test_names, output_files, legend_names, fontsize=fontsize)

    output_files = ['final_plots/hopper/hop_seed_' + name for name in test_names]
    output_files.append('final_plots/hopper/compare_all_seed')
    file_names = [data_dir + 'hop_0adv_concat1_seed_lv0p9_lr0005/',
                  data_dir + 'hop_1adv_concat1_seed_str0p25_lv0p0_lr0005/',
                  data_dir + 'hop_0adv_concat10_seed_dr_lv0p5_lr00005/',
                  data_dir + 'hop_5adv_concat1_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/']
    legend_names = ['0 Adversary', 'RARL, No Memory', 'DR, Memory', '5 Adversary, No Memory']
    titles = ['blah' for i in range(len(test_names))] + ['Average score across test set on 10 seeds']
    with open('final_plots/hopper/test_to_str.txt', 'w') as file:
        for a, b in zip([string.ascii_uppercase[i] for i in range(len(test_names))], test_names):
            file.write(a + ' ' + b + '\n')
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3800], titles=titles,
                      fontsize=fontsize)
    output_files[-1] = 'final_plots/hopper/compare_all_seed_std'
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 5000], titles=titles,
                      fontsize=fontsize, use_std=True)

    # generate the test set maps for the ablations
    output_files = ['final_plots/hopper/hop_seed_ablate' + name for name in test_names]
    output_files.append('final_plots/hopper/compare_all_ablate')
    file_names = [data_dir + 'hop_5adv_concat10_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/',
                  data_dir + 'hop_5adv_concat1_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/',
                  data_dir + 'hop_5adv_concat10_seed_str0p25_norew_lv0p5_lr00005/',
                  data_dir + 'hop_5adv_concat1_seed_str0p25_norew_lv0p5_lr00005/'
                  ]
    legend_names = ['Memory, Reward Range', 'No Memory, Reward Range',
                    'Memory, No Reward Range', 'No Memory, No Reward Range']
    titles = ['blah' for i in range(len(test_names))] + ['Average score across test set on 10 seeds']
    with open('final_plots/hopper/ablate_test_to_str.txt', 'w') as file:
        for a, b in zip([string.ascii_uppercase[i] for i in range(len(test_names))], test_names):
            file.write(a + ' ' + b + '\n')
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 4000], titles=titles,
                      fontsize=fontsize)

    # ablation on domain randomization

    output_files = ['final_plots/hopper/hop_seed_dr_' + name for name in test_names]
    output_files.append('final_plots/hopper/compare_all_DR')
    file_names = [data_dir + 'hop_0adv_concat10_seed_dr_lv0p5_lr00005/',
                  data_dir + 'hop_0adv_concat1_seed_dr_lv0p9_lr0005/',]
    legend_names = ['DR, Memory', 'DR, no memory']
    titles = ['blah' for i in range(len(test_names))] + ['Average score across test set on 10 seeds']
    with open('final_plots/hopper/dr_ablate_test_to_str.txt', 'w') as file:
        for a, b in zip([string.ascii_uppercase[i] for i in range(len(test_names))], test_names):
            file.write(a + ' ' + b + '\n')
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3200], titles=titles,
                      fontsize=fontsize)