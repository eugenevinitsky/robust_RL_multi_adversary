#!/usr/bin/env bash

## 2D
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d2_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test2
##
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
### 4D
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.1 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d4_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.1 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.1 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
### 6D
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d6_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.066 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.066 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d6_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.066 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test9
#
### 8D
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d8_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.05 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test10
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.05 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test11
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8_conc25_h25_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.05 --num_concat_states 25 --horizon 25" \
#--start --stop --tmux --cluster-name=ev_lin_test12


# 1/30 experiments. We don't have to rerun the domain randomization ones since those are nice and frozen. We will have
# to recompute their scores though as the graphs are wrong! Note that thea adversaries were too strong here
#
## 2d experiments
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d2_conc25_h25 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 2d experiments l2
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --l2_reward --l2_memory \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25_l2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 --l2_reward --l2_memory \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d2_conc25_h25_l2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 4d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc25_h25 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d4_conc25_h25 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 4d l2
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --l2_reward --l2_memory \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc25_h25_l2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 --l2_reward --l2_memory \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d4_conc25_h25_l2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## 8d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test9
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test10
#
## 8d l2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --l2_reward --l2_memory \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_l2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test11
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --l2_reward --l2_memory \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_l2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test12
#
## 8d with low reward of -150
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -150 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low150 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test13
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --low_reward -150 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_low150 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test14
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --l2_reward --l2_memory --low_reward -150 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_l2_low150 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test15
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --l2_reward --l2_memory  --low_reward -150\
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_l2_low150 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test116
#

# second round of experiments

# 2d experiments with low of 100
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -100 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25_low100_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test1

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 --low_reward -100 \
--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d2_conc25_h25_low100_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test2

# 2d experiments with low reward of -200
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25_low200_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test3

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d2_conc25_h25_low200_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test4

# 8d with low reward of -400
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low400_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test5

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --low_reward -400 \
--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_low400_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test6

# 8d with low reward of -600
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test7

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test8

# 2d dr experiments
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc25_h25_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test9

# 8d dr experiments
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8_conc25_h25_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test10

