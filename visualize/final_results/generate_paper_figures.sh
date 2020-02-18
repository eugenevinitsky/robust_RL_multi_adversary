# generate transfer plots for 2 arms
# python visualize/bandit/bandit_result_plotting.py visualize/final_results/final_plots/bandit/2\ arms --num_arms 2 --output_file_name 2arm --group_seeds mean --spacing 7

# generate transfer plots for 5 arms
# python visualize/bandit/bandit_result_plotting.py visualize/final_results/final_plots/bandit/5\ arms --num_arms 5 --output_file_name 5arm --group_seeds mean --spacing 6

# generate adversary diversity examples:

python visualize/bandit/visualize_adversaries.py visualize/final_results/final_plots/bandit/bandit_arm2_6adv_04_02_162058/PPO_0_2020-02-05_00-26-14wxw20oeq 500 --num_samples 50 --output_dir visualize/final_results/final_plots/bandit/



###########
# Transfer tests:
# Avg Transfer Test Perf:
# A: spread_high_std
# B: cluster_high_std
# C: one_good_boi
# D: needle_in_haystack
# E: hard