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

# 2d experiments with low of -100
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -100 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25_low100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 --low_reward -100 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d2_conc25_h25_low100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 2d experiments with low reward of -200
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc25_h25_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 10 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d2_conc25_h25_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 8d with low reward of -400
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low400_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_low400_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 8d with low reward of -600
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 10 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 2 --num_adv_rews 5 --exp_title linear_5adv_2per_d8_conc25_h25_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## 2d dr experiments
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc25_h25_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test9
#
## 8d dr experiments
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8_conc25_h25_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test10
#
## 8d with low reward of -600 and lower action cost coeff
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -600 --action_cost_coeff 5.0 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low600_coeff_5r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test11
#
## 8d with low reward of -400 and lower action cost coeff
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 --action_cost_coeff 5.0 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc25_h25_low400_coeff_5r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test13
#
## 8d dr experiments and lower action cost coeff
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 60000 --num_cpus 4 --advs_per_strength 0 --action_cost_coeff 5.0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8_conc25_h25_coeff5_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 8 --multi_node --num_concat_states 25 --horizon 25 --scaling -0.8 --adv_strength 0.05 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test12

############################################################################################################################################
# 2/01/2020
############################################################################################################################################
# 2d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
## 2d experiments with low of -400, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low400_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 2d experiments with low of -600, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_5adv_d2_conc100_h100_low600_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
## 2d experiments with low of 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d2_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 4d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
## 4d experiments with low of -400, 4 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low400_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 4d experiments with low of -800, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 1 --low_reward -800 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_5adv_d4_conc100_h100_low800_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
## 4d experiments with 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d4_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## domain randomization in 2d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test9
#
## domain randomization in 4d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test10

############################################################################################################################################
# 2/02/2020
# Same experiments as above but an additional 6d experiment to illustrate the breakdown even more cleanly
############################################################################################################################################
# 2d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
## 2d experiments with low of -400, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low400_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 2d experiments with low of -600, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_5adv_d2_conc100_h100_low600_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
## 2d experiments with low of 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d2_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 4d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
## 4d experiments with low of -400, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low400_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 4d experiments with low of -800, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 1 --low_reward -800 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_5adv_d4_conc100_h100_low800_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
## 4d experiments with 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d4_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## 6d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test9
#
## 6d experiments with low of -400, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc100_h100_low400_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test10
#
## 6d experiments with low of -800, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 1 --low_reward -800 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_5adv_d6_conc100_h100_low800_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test11
#
## 6d experiments with 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d6_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test12
#
### domain randomization in 2d
##ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
##--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
##--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r1 --run_transfer_tests \
##--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
##--start --stop --tmux --cluster-name=ev_lin_test13
##
### domain randomization in 4d
##ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
##--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 0 \
##--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r1 --run_transfer_tests \
##--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
##--start --stop --tmux --cluster-name=ev_lin_test114
##
### domain randomization in 6d
##ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
##--num_iters 500 --train_batch_size 30000 --num_cpus 4 --advs_per_strength 0 \
##--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d6_conc100_h100_r1 --run_transfer_tests \
##--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
##--start --stop --tmux --cluster-name=ev_lin_test115

# 2-04 experiments
# domain randomization in 2d where we sample eigenvalues instead of uniformly
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d2_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4 \
#--eigval_rand" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
## domain randomization in 4d where we sample eigenvalues instead of uniformly
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d4_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4 \
#--eigval_rand" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## domain randomization in 6d where we sample eigenvalues instead of uniformly
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d6_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4 \
#--eigval_rand" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
## domain randomization in 2d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## domain randomization in 4d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
## domain randomization in 6d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d6_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.0666 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test6

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d2_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.4 --agent_strength 0.4 \
#--eigval_rand" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
## domain randomization in 4d where we sample eigenvalues instead of uniformly
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d4_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.4 --agent_strength 0.4 \
#--eigval_rand" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## domain randomization in 6d where we sample eigenvalues instead of uniformly
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d6_conc100_h100_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.4 --agent_strength 0.4 \
#--eigval_rand" \
#--start --stop --tmux --cluster-name=ev_lin_test9

############################## 2/04. Repeat of 2 / 02 experiments but with better ranges and no bugs

# 2d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
## 2d experiments with low of -400, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 5 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low400_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 2d experiments with low of -600, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 5 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low600_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
## 2d experiments with low of 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d2_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 4d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
## 4d experiments with low of -1000, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -1000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low4=1000_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 4d experiments with low of -4000, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -4000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low4000_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test16
#
## 4d experiments with 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d4_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## 6d experiments with low of -200, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc100_h100_low200_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.066 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test17
#
## 6d experiments with low of -1000, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -1000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc100_h100_low1000_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.066 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test10
#
## 6d experiments with low of -6000, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -6000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc100_h100_low6000_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.066 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test11
#
## 6d experiments with 1 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d6_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.066 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test12
#
## domain randomization in 2d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test13
#
## domain randomization in 4d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test14
#
## domain randomization in 6d
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d6_conc100_h100_r1 --run_transfer_tests \
#--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.066 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test15


# Round 2 experiments 2/04
# 2d experiments with low of -200, 10 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 10 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d2_conc100_h100_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test21
#
## 2d experiments with low of -400, 10 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 10 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d2_conc100_h100_low400_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test22
#
## 2d experiments with low of -600, 10 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 10 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d2_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test23
#
## 4d experiments with low of -600, 5 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test24
#
## 4d experiments with low of -200, 10 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d4_conc100_h100_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test25
#
## 4d experiments with low of -600, 10 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test26
#
## 4d experiments with low of -1000, 10 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 --low_reward -1000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d4_conc100_h100_low1000_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test27
#
## 2d experiments with low of -200, 20 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 20 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d2_conc100_h100_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test28
#
## 2d experiments with low of -400, 20 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 20 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d2_conc100_h100_low400_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test29
#
## 2d experiments with low of -600, 20 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 20 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d2_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test30
#
#
## 4d experiments with low of -200, 20 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d4_conc100_h100_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test31
#
## 4d experiments with low of -600, 20 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test32
#
#
## 4d experiments with low of -1000, 20 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 --low_reward -1000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d4_conc100_h100_low1000_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test33
#
#
## 2d experiments with low of -200, 40 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 40 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 40 --exp_title linear_40adv_d2_conc100_h100_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test34
#
## 2d experiments with low of -400, 40 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 40 --low_reward -400 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 40 --exp_title linear_40adv_d2_conc100_h100_low400_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test35
#
## 2d experiments with low of -600, 40 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 40 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 40 --exp_title linear_40adv_d2_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test36
#
#
## 4d experiments with low of -200, 40 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 40 --low_reward -200 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 40 --exp_title linear_40adv_d4_conc100_h100_low200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test37
#
## 4d experiments with low of -600, 40 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 40 --low_reward -600 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 40 --exp_title linear_40adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test38
#
#
## 4d experiments with low of -1000, 40 adv
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 40 --low_reward -1000 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 40 --exp_title linear_40adv_d4_conc100_h100_low1000_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
#--start --stop --tmux --cluster-name=ev_lin_test39


######################################################################
# 4/12 SOTA tests
# Pure domain randomization
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d3_h200_r2 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.4 --agent_strength 0.4 \
#--eigval_rand --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
## 3d experiments w/ 1 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_RARL_d3_conc100_h200 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 3d experiments w/ 5 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 5 --num_adv_rews 1 --exp_title linear_RARL_5adv_d3_conc100_h200 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
## 3d experiments w/ 10 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 \
#--num_adv_strengths 1 --advs_per_rew 10 --num_adv_rews 1 --exp_title linear_RARL_10adv_d3_conc100_h200 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 3d experiments w/ 20 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 \
#--num_adv_strengths 1 --advs_per_rew 20 --num_adv_rews 1 --exp_title linear_RARL_10adv_d3_conc100_h200 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
## 3d experiments w/ 5 DMALT adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_conc100_h200 --run_transfer_tests \
#--reward_range --low_reward -400 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 3d experiments w/ 10 DMALT adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_conc100_h200 --run_transfer_tests \
#--reward_range --low_reward -400 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
## 3d experiments w/ 20 DMALT adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_conc100_h200 --run_transfer_tests \
#--reward_range --low_reward -400 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test8

#######################################################################
## 4/13 SOTA tests w/ more sensible reward range
#
## Pure domain randomization
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d3_h200_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.4 --agent_strength 0.4 \
#--eigval_rand --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
## 3d experiments w/ 1 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 1 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_RARL_d3_conc100_h200_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
## 3d experiments w/ 5 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 5 --num_adv_rews 1 --exp_title linear_RARL_5adv_d3_conc100_h200_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test3
#
## 3d experiments w/ 10 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 10 \
#--num_adv_strengths 1 --advs_per_rew 10 --num_adv_rews 1 --exp_title linear_RARL_10adv_d3_conc100_h200_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test4
#
## 3d experiments w/ 20 RARL adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 20 \
#--num_adv_strengths 1 --advs_per_rew 20 --num_adv_rews 1 --exp_title linear_RARL_10adv_d3_conc100_h200_r4 --run_transfer_tests \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test5
#
### 3d experiments w/ 5 DMALT adversaries and low of -10
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_h200_low10_r4 --run_transfer_tests \
#--reward_range --low_reward -10 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test6
#
## 3d experiments w/ 10 DMALT adversaries and low of -10
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 10 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_h200_low10_r4 --run_transfer_tests \
#--reward_range --low_reward -10 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test7
#
## 3d experiments w/ 20 DMALT adversaries and low of -10
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 20 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_h200_low10_r4 --run_transfer_tests \
#--reward_range --low_reward -10 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test8
#
## 3d experiments w/ 5 DMALT adversaries and low of -100
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_h200_low100_r4 --run_transfer_tests \
#--reward_range --low_reward -100 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test9
#
## 3d experiments w/ 10 DMALT adversaries and low of -100
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 10 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_h200_low100_r4 --run_transfer_tests \
#--reward_range --low_reward -100 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test10
#
## 3d experiments w/ 20 DMALT adversaries and low of -100
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 20 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_h200_low100_r4 --run_transfer_tests \
#--reward_range --low_reward -100 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test11
#
### 3d experiments w/ 5 DMALT adversaries and low of -1000
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 5 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_h200_low1000_r4 --run_transfer_tests \
#--reward_range --low_reward -1000 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test12
#
## 3d experiments w/ 10 DMALT adversaries and low of -1000
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 10 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_h200_low1000_r4 --run_transfer_tests \
#--reward_range --low_reward -1000 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test13
#
## 3d experiments w/ 20 DMALT adversaries and low of -1000
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 250 --train_batch_size 300000 --num_cpus 16 --advs_per_strength 20 \
#--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_h200_low1000_r4 --run_transfer_tests \
#--reward_range --low_reward -1000 \
#--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
#--start --stop --tmux --cluster-name=ev_lin_test14

######################################################################
# 4/14 SOTA tests w/ more sensible reward range AND changes to how the K_init is constructed that make the problem
# a lot harder.

# Pure domain randomization
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 60000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d3_h200_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.4 --agent_strength 0.4 \
--eigval_rand --regret" \
--start --stop --tmux --cluster-name=ev_lin_test1

# 3d experiments w/ 1 RARL adversary
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 1 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_RARL_d3_conc100_h200_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test2

# 3d experiments w/ 5 RARL adversary
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 \
--num_adv_strengths 1 --advs_per_rew 5 --num_adv_rews 1 --exp_title linear_RARL_5adv_d3_conc100_h200_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test3

# 3d experiments w/ 10 RARL adversary
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 \
--num_adv_strengths 1 --advs_per_rew 10 --num_adv_rews 1 --exp_title linear_RARL_10adv_d3_conc100_h200_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test4

# 3d experiments w/ 20 RARL adversary
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 \
--num_adv_strengths 1 --advs_per_rew 20 --num_adv_rews 1 --exp_title linear_RARL_10adv_d3_conc100_h200_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.133 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test5

## 3d experiments w/ 5 DMALT adversaries and low of -10
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_h200_low10_r4 --run_transfer_tests \
--reward_range --low_reward -10 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test6

# 3d experiments w/ 10 DMALT adversaries and low of -10
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_h200_low10_r4 --run_transfer_tests \
--reward_range --low_reward -10 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test7

# 3d experiments w/ 20 DMALT adversaries and low of -10
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_h200_low10_r4 --run_transfer_tests \
--reward_range --low_reward -10 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test8

# 3d experiments w/ 5 DMALT adversaries and low of -100
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_h200_low100_r4 --run_transfer_tests \
--reward_range --low_reward -100 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test9

# 3d experiments w/ 10 DMALT adversaries and low of -100
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_h200_low100_r4 --run_transfer_tests \
--reward_range --low_reward -100 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test10

# 3d experiments w/ 20 DMALT adversaries and low of -100
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_h200_low100_r4 --run_transfer_tests \
--reward_range --low_reward -100 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test11

## 3d experiments w/ 5 DMALT adversaries and low of -1000
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_DAMLT_5adv_d3_h200_low1000_r4 --run_transfer_tests \
--reward_range --low_reward -1000 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test12

# 3d experiments w/ 10 DMALT adversaries and low of -1000
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_DAMLT_10adv_d3_h200_low1000_r4 --run_transfer_tests \
--reward_range --low_reward -1000 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test13

# 3d experiments w/ 20 DMALT adversaries and low of -1000
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_DAMLT_20adv_d3_h200_low1000_r4 --run_transfer_tests \
--reward_range --low_reward -1000 \
--grid_search --use_s3 --dim 3 --multi_node --horizon 200 --scaling -0.8 --adv_strength 0.1333 --agent_strength 0.4 --regret" \
--start --stop --tmux --cluster-name=ev_lin_test14
