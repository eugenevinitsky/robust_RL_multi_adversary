import os
import string

import matplotlib
import matplotlib.cm as cm
font = {'family' : 'normal',
        'size' : 12}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import numpy as np

from visualize.linear_env.test_eigenvals import plot_eigenvals


def generate_bar_plots(file_list, title, file_name, x_title=None, y_title=None, open_cmd=lambda x: np.load(x), legend_rule=None):
    plt.figure()
    if x_title:
        plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
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
    ax = plt.bar(np.arange(len(file_list)), mean_list, yerr=std_list, color=colors, capsize=3)
    plt.legend(ax, legend_titles)
    plt.xticks([])
    # plt.xticks(np.arange(len(file_list)), tick_titles)
    # plt.tick_params(bottom=False)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_across_folders(folder_list, test_names, file_names, legend_names, open_cmd=lambda x: np.loadtxt(x)):
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
    plt.xticks()
    plt.savefig(file_names[-1])
    plt.close()


def plot_across_seeds(outer_folder_list, test_names, file_names, legend_names, num_seeds, ylabel='Avg. Score',
                      open_cmd=lambda x: np.loadtxt(x), yaxis=None, titles=[]):
    test_results = np.zeros((len(test_names), len(outer_folder_list)))
    colors = cm.rainbow(np.linspace(0, 1, len(outer_folder_list)))
    for i, folder in enumerate(outer_folder_list):
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for file in filenames:
                found_idx = [test_name in file and '.png' not in file for test_name in test_names]
                if np.sum(found_idx) > 0:
                    idx = np.where(np.array(found_idx) > 0)
                    test_idx = idx[0][0]
                    test_results[test_idx, i] += np.mean(open_cmd(os.path.join(dirpath, file)))

    for i, test in enumerate(test_names):
        plt.figure()
        ax = plt.bar(np.arange(len(outer_folder_list)), test_results[i, :] / num_seeds, color=colors, capsize=3)
        plt.tight_layout()
        plt.legend(ax, legend_names)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.title(titles[i])
        plt.savefig(file_names[i])

    # Now generate a plot for all of them
    dist = 7
    plt.figure()
    for i, test in enumerate(test_names):
        ax = plt.bar(np.arange(len(outer_folder_list)) + dist * i, test_results[i, :] / num_seeds, color=colors, capsize=3)
        plt.tight_layout()
        plt.legend(ax, legend_names)
    plt.xticks([int(len(outer_folder_list) / 2) - 0.5 + (dist) * i for i in range(len(test_names))],
               string.ascii_uppercase[0:len(test_names)])
    plt.ylabel(ylabel)
    if yaxis:
        plt.ylim(yaxis)
    plt.title(titles[-1])
    plt.tight_layout()

    plt.savefig(file_names[-1])



if __name__ == '__main__':
    ###################################################################################################################
    ######################################### LINEAR SYSTEMS #########################################################
    ###################################################################################################################
    # Generate eigenvalue plots for the linear env
    plot_eigenvals(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_plots/linear_env'))

    ################################################### 2 DIMENSIONS #################################################
    # Generate the bar plots for the linear system, 2d, base score
    data_dir = '/Users/eugenevinitsky/Desktop/Research/Data/sim2real/transfer_results/linear_env/02-02-2020/'
    file_list = [data_dir + 'linear_1adv_d2_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-40-21zsnsvemv/'
                            'linear_1adv_d2_conc100_h100_r1_base_sweep_rew',
                 data_dir + 'linear_5adv_d2_conc100_h100_low600_r1/'
                            'PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-39-06smbqpg8u/linear_5adv_d2_conc100_h100_low600_r1_base_sweep_rew',
                 data_dir + 'linear_dr_d2_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-02_23-51-38zfkfbafv/linear_dr_d2_conc100_h100_r1_base_sweep_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for base system, Dimension 2'
    file_name = 'final_plots/linear_env/base_regret_2d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    # Generate the bar plots for the linear system, 2d, base domain randomization score
    file_list = [data_dir + 'linear_1adv_d2_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-40-21zsnsvemv/'
                            'linear_1adv_d2_conc100_h100_r1_domain_rand_rew',
                 data_dir + 'linear_5adv_d2_conc100_h100_low600_r1/'
                            'PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-39-06smbqpg8u/linear_5adv_d2_conc100_h100_low600_r1_domain_rand_rew',
                 data_dir + 'linear_dr_d2_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-02_23-51-38zfkfbafv/linear_dr_d2_conc100_h100_r1_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for domain randomization, Dimension 2'
    file_name = 'final_plots/linear_env/drand_regret_2d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    # Generate the bar plots for the linear system, 2d, hard domain randomization score
    file_list = [data_dir + 'linear_1adv_d2_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-40-21zsnsvemv/'
                            'linear_1adv_d2_conc100_h100_r1_hard_domain_rand_rew',
                 data_dir + 'linear_5adv_d2_conc100_h100_low600_r1/'
                            'PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-39-06smbqpg8u/linear_5adv_d2_conc100_h100_low600_r1_hard_domain_rand_rew',
                 data_dir + 'linear_dr_d2_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-02_23-51-38zfkfbafv/linear_dr_d2_conc100_h100_r1_hard_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for unstable systems, Dimension 2'
    file_name = 'final_plots/linear_env/hard_regret_2d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    ################################################### 4 DIMENSIONS #################################################

    # Generate the bar plots for the linear system, 4d, base score
    file_list = [data_dir + 'linear_1adv_d4_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-45-28uumkborq/'
                            'linear_1adv_d4_conc100_h100_r1_base_sweep_rew',
                 data_dir + 'linear_5adv_d4_conc100_h100_low800_r1/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-02_23-44-13mp7f36hf/linear_5adv_d4_conc100_h100_low800_r1_base_sweep_rew',
                 data_dir + 'linear_dr_d4_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-03_00-11-29_98endso/linear_dr_d4_conc100_h100_r1_base_sweep_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for base system, Dimension 4'
    file_name = 'final_plots/linear_env/base_regret_4d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    # Generate the bar plots for the linear system, 4d, base domain randomization score
    file_list = [data_dir + 'linear_1adv_d4_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-45-28uumkborq/'
                            'linear_1adv_d4_conc100_h100_r1_domain_rand_rew',
                 data_dir + 'linear_5adv_d4_conc100_h100_low800_r1/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-02_23-44-13mp7f36hf/linear_5adv_d4_conc100_h100_low800_r1_domain_rand_rew',
                 data_dir + 'linear_dr_d4_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-03_00-11-29_98endso/linear_dr_d4_conc100_h100_r1_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for domain randomization, Dimension 4'
    file_name = 'final_plots/linear_env/drand_regret_4d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    # Generate the bar plots for the linear system, 4d, hard domain randomization score
    file_list = [data_dir + 'linear_1adv_d4_conc100_h100_r1/PPO_3_lambda=0.5,lr=5e-05_2020-02-02_23-45-28uumkborq/'
                            'linear_1adv_d4_conc100_h100_r1_hard_domain_rand_rew',
                 data_dir + 'linear_5adv_d4_conc100_h100_low800_r1/'
                            'PPO_4_lambda=0.9,lr=5e-05_2020-02-02_23-44-13mp7f36hf/linear_5adv_d4_conc100_h100_low800_r1_hard_domain_rand_rew',
                 data_dir + 'linear_dr_d4_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-03_00-11-29_98endso/linear_dr_d4_conc100_h100_r1_hard_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for unstable systems, Dimension 4'
    file_name = 'final_plots/linear_env/hard_regret_4d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    ################################################### 6 DIMENSIONS #################################################

    # Generate the bar plots for the linear system, 6d, base score
    file_list = [data_dir + 'linear_1adv_d6_conc100_h100_r1/PPO_0_lambda=0.5,lr=0.0005_2020-02-02_23-50-25udhwqfrx/'
                            'linear_1adv_d6_conc100_h100_r1_base_sweep_rew',
                 data_dir + 'linear_5adv_d6_conc100_h100_low800_r1/'
                            'PPO_2_lambda=1.0,lr=0.0005_2020-02-02_23-49-10gwv8ac_z/linear_5adv_d6_conc100_h100_low800_r1_base_sweep_rew',
                 data_dir + 'linear_dr_d6_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-02_23-52-592m4c6sii/linear_dr_d6_conc100_h100_r1_base_sweep_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for base system, Dimension 6'
    file_name = 'final_plots/linear_env/base_regret_6d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    # Generate the bar plots for the linear system, 6d, base domain randomization score
    file_list = [data_dir + 'linear_1adv_d6_conc100_h100_r1/PPO_0_lambda=0.5,lr=0.0005_2020-02-02_23-50-25udhwqfrx/'
                            'linear_1adv_d6_conc100_h100_r1_domain_rand_rew',
                 data_dir + 'linear_5adv_d6_conc100_h100_low800_r1/'
                            'PPO_2_lambda=1.0,lr=0.0005_2020-02-02_23-49-10gwv8ac_z/linear_5adv_d6_conc100_h100_low800_r1_domain_rand_rew',
                 data_dir + 'linear_dr_d6_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-02_23-52-592m4c6sii/linear_dr_d6_conc100_h100_r1_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for domain randomization, Dimension 6'
    file_name = 'final_plots/linear_env/drand_regret_6d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    # Generate the bar plots for the linear system, 6d, hard domain randomization score
    file_list = [data_dir + 'linear_1adv_d6_conc100_h100_r1/PPO_0_lambda=0.5,lr=0.0005_2020-02-02_23-50-25udhwqfrx/'
                            'linear_1adv_d6_conc100_h100_r1_hard_domain_rand_rew',
                 data_dir + 'linear_5adv_d6_conc100_h100_low800_r1/'
                            'PPO_2_lambda=1.0,lr=0.0005_2020-02-02_23-49-10gwv8ac_z/linear_5adv_d6_conc100_h100_low800_r1_hard_domain_rand_rew',
                 data_dir + 'linear_dr_d6_conc100_h100_r1/'
                            'PPO_5_lambda=1.0,lr=5e-05_2020-02-02_23-52-592m4c6sii/linear_dr_d6_conc100_h100_r1_hard_domain_rand_rew']
    y_title = 'Total regret over 100 steps'
    legend_titles = ['1 Adversary', '5 Adversaries', 'Domain randomization']
    title = 'Accumulated regret for unstable systems, Dimension 6'
    file_name = 'final_plots/linear_env/hard_regret_6d.png'
    generate_bar_plots(file_list, title, file_name, None, y_title)

    ###################################################################################################################
    ######################################### HOPPER #########################################################
    ###################################################################################################################

    # generate the relevant heatmaps that will go into the paper

    # generate the bar charts comparing 0 adv, dr, and 5 adv for hopper

    # generate the test set maps for the best validation set result
    data_dir = '/Users/eugenevinitsky/Desktop/Research/Data/sim2real/transfer_results/adv_robust/02-02-2020/'
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
    output_files = ['hopper/hop' + name for name in test_names]
    output_files.append('hopper/compare_all')

    file_names = [data_dir + 'hop_0adv_concat1_seed_lv0p9_lr0005/PPO_2_seed=2_2020-02-02_22-49-49107ecvph/',
                  data_dir + 'hop_1adv_concat1_seed_str0p25_lv0p0_lr0005/PPO_9_seed=9_2020-02-02_22-53-39u2gwqrg8',
                  data_dir + 'hop_0adv_concat10_seed_dr_lv0p5_lr00005/PPO_0_seed=0_2020-02-02_23-06-31qtb9bzqs',
                  data_dir + 'hop_5adv_concat10_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/PPO_0_seed=0_2020-02-02_22-56-06h0gae7mk']
    legend_names = ['0 Adversary', '1 Adversary No Memory', 'DR', '5 Adv Memory']
    plot_across_folders(file_names, test_names, output_files, legend_names)

    output_files = ['hopper/hop_seed_' + name for name in test_names]
    output_files.append('hopper/compare_all_seed')
    file_names = [data_dir + 'hop_0adv_concat1_seed_lv0p9_lr0005/',
                  data_dir + 'hop_1adv_concat1_seed_str0p25_lv0p0_lr0005/',
                  data_dir + 'hop_0adv_concat10_seed_dr_lv0p5_lr00005/',
                  data_dir + 'hop_5adv_concat10_seed_str0p25rew_l1000_h3500_lv0p9_lr0005/']
    legend_names = ['0 Adversary', '1 Adversary, No Memory', 'DR', '5 Adversary, Memory']
    titles = ['blah' for i in range(len(test_names))] + ['Average score across test set on 10 seeds']
    plot_across_seeds(file_names, test_names, output_files, legend_names, num_seeds=10, yaxis=[0, 3200], titles=titles)
