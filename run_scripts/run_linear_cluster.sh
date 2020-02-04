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
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d2_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4 \
--eigval_rand" \
--start --stop --tmux --cluster-name=ev_lin_test1

# domain randomization in 4d where we sample eigenvalues instead of uniformly
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d4_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4 \
--eigval_rand"
--start --stop --tmux --cluster-name=ev_lin_test2

# domain randomization in 6d where we sample eigenvalues instead of uniformly
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_er_d6_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4\
--eigval_rand" \
--start --stop --tmux --cluster-name=ev_lin_test3

# domain randomization in 2d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test4

# domain randomization in 4d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test5

# domain randomization in 6d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d6_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 6 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.0666 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test6