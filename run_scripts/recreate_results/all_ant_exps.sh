#!/usr/bin/env bash

################################################################################################
############################ HYPERPARAM SWEEP ################################################
# Uncomment these experiments to run the hyperparameter search. If all you want is to run the seed sweep
# look at the section below

# 0 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name ant --exp_title ant_0adv_concat1_grid --num_cpus 10 --run_transfer_tests --multi_node --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test1
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/visualize/mujoco/cluster_transfer_tests.py ant_0adv_concat1_grid 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest1
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name ant --exp_title ant_0adv_concat1_seed --num_cpus 10 --run_transfer_tests --multi_node --seed_search --adv_all_actions --concat_actions --lambda_val 0.5 --lr .0005" --start --stop --tmux --cluster-name=yd_ant_stest1

# 1 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name ant --exp_title ant_1adv_concat1_grid_str0p1 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.1 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test2
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name ant --exp_title ant_1adv_concat1_grid_str0p5 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.5 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test3

ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/visualize/mujoco/cluster_transfer_tests.py ant_1adv_concat1_grid_str0p25 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest2
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name ant --exp_title ant_1adv_concat1_seed_str0p15 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --seed_search --adv_all_actions --concat_actions --lambda_val 0.5 --lr .0005" --start --stop --tmux --cluster-name=yd_ant_stest2

# 3 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name ant --exp_title ant_3adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test6
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/visualize/mujoco/cluster_transfer_tests.py ant_5adv_concat1_grid_str0p25_norew 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest3
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name ant --exp_title ant_3adv_concat1_seed_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --seed_search --adv_all_actions --concat_actions  --lambda_val 0.5 --lr .0005" --start --stop --tmux --cluster-name=yd_ant_stest6


# 5 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name ant --exp_title ant_5adv_concat1_grid_str0p2_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.2 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test6
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name ant --exp_title ant_5adv_concat1_grid_str0p5_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.5 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test7

ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/visualize/mujoco/cluster_transfer_tests.py ant_5adv_concat1_grid_str0p25_norew 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest3
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name ant --exp_title ant_5adv_concat1_seed_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --seed_search --adv_all_actions --concat_actions  --lambda_val 0.5 --lr .0005" --start --stop --tmux --cluster-name=yd_ant_stest6


# Domain randomization experiment w/ grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name ant --exp_title ant_0adv_concat1_grid_dr --num_cpus 10 --run_transfer_tests --multi_node --grid_search --domain_randomization --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_ant_test9
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/visualize/mujoco/cluster_transfer_tests.py ant_0adv_concat1_grid_dr 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest4
#Seed Seep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name ant --exp_title ant_0adv_concat1_seed_dr --num_cpus 10 --run_transfer_tests --multi_node --seed_search --domain_randomization --adv_all_actions --concat_actions --lambda_val 0.9 --lr .00005" --start --stop --tmux --cluster-name=yd_ant_stest9


