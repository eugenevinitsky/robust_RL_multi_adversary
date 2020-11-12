#!/usr/bin/env bash

################################################################################################
############################ HYPERPARAM SWEEP ################################################
# Uncomment these experiments to run the hyperparameter search. If all you want is to run the seed sweep
# look at the section below

# 0 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name hopper --exp_title hop_0adv_concat1_grid --num_cpus 10 --run_transfer_tests --multi_node --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test1
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name hopper --exp_title hop_0adv_concat1_seed --num_cpus 10 --run_transfer_tests --multi_node --seed_search --adv_all_actions --concat_actions --lambda_val 1.0 --lr .00005" --start --stop --tmux --cluster-name=yd_hop_stest1

# 1 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name hopper --exp_title hop_1adv_concat1_grid_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test2
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name hopper --exp_title hop_1adv_concat1_seed_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 1.0 --lr .0005" --start --stop --tmux --cluster-name=yd_hop_stest2

# 2 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 2 --advs_per_rew 1 --num_adv_rews 2 --use_s3 --env_name hopper --exp_title hop_2adv_concat1_grid_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test2
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 2 --advs_per_rew 1 --num_adv_rews 2 --use_s3 --env_name hopper --exp_title hop_2adv_concat1_seed_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 0.5 --lr .00005" --start --stop --tmux --cluster-name=yd_hop_stest2


# 3 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name hopper --exp_title hop_3adv_concat1_grid_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test3
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name hopper --exp_title hop_3adv_concat1_seed_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 0.5 --lr .0005" --start --stop --tmux --cluster-name=yd_hop_stest3


# 5 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name hopper --exp_title hop_5adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test4
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name hopper --exp_title hop_5adv_concat1_seed_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 0.5 --lr .0005" --start --stop --tmux --cluster-name=yd_hop_stest4

# 7 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 7 --advs_per_rew 1 --num_adv_rews 7 --use_s3 --env_name hopper --exp_title hop_7adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test5
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 7 --advs_per_rew 1 --num_adv_rews 7 --use_s3 --env_name hopper --exp_title hop_7adv_concat1_seed_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_hop_stest5

# 9 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 9 --advs_per_rew 1 --num_adv_rews 9 --use_s3 --env_name hopper --exp_title hop_9adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test6
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 9 --advs_per_rew 1 --num_adv_rews 9 --use_s3 --env_name hopper --exp_title hop_9adv_concat1_seed_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 1.0 --lr .00005" --start --stop --tmux --cluster-name=yd_hop_stest6

# 11 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 11 --advs_per_rew 1 --num_adv_rews 11 --use_s3 --env_name hopper --exp_title hop_11adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test7
#Seed Sweep
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 11 --advs_per_rew 1 --num_adv_rews 11 --use_s3 --env_name hopper --exp_title hop_11adv_concat1_seed_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --seed_search --adv_all_actions --concat_actions --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_hop_stest7

# 13 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 13 --advs_per_rew 1 --num_adv_rews 13 --use_s3 --env_name hopper --exp_title hop_13adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test8

# 15 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 15 --advs_per_rew 1 --num_adv_rews 15 --use_s3 --env_name hopper --exp_title hop_15adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_hop_test9
