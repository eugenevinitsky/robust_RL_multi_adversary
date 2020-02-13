#!/usr/bin/env bash

################################################################################################
############################ HYPERPARAM SWEEP ################################################
# Uncomment these experiments to run the hyperparameter search. We do not perform a seed search# as these experiments are purely illustratory


# 2d experiments 1 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 1 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d2_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test1

# 2d experiments with low of -200, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 5 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low200_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test2

# 2d experiments with low of -400, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 5 --low_reward -400 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low400_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test3

# 2d experiments with low of -600, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 5 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc100_h100_low600_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test4

# 2d experiments with low of -200, 10 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 10 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d2_conc100_h100_low200_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test5

# 2d experiments with low of -400, 10 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 10 --low_reward -400 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d2_conc100_h100_low400_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test6

# 2d experiments with low of -600, 10 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 10 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d2_conc100_h100_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test7

# 2d experiments with low of -200, 20 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 20 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d2_conc100_h100_low200_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test8

# 2d experiments with low of -400, 20 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 20 --low_reward -400 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d2_conc100_h100_low400_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test9

# 2d experiments with low of -600, 20 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 20 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d2_conc100_h100_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test10

# 4d experiments with low of -200, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low200_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test11

# 4d experiments with low of -1000, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -1000 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low4=1000_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test12

# 4d experiments with low of -4000, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -4000 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low4000_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test13

# 4d experiments with 1 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 1 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d4_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test14

# domain randomization in 2d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test15

# domain randomization in 4d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test16

# 4d experiments with low of -600, 5 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 5 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test17

# 4d experiments with low of -200, 10 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d4_conc100_h100_low200_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test18

# 4d experiments with low of -600, 10 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test19

# 4d experiments with low of -1000, 10 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 10 --low_reward -1000 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 10 --exp_title linear_10adv_d4_conc100_h100_low1000_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test20

# 4d experiments with low of -200, 20 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 --low_reward -200 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d4_conc100_h100_low200_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test21

# 4d experiments with low of -600, 20 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 --low_reward -600 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d4_conc100_h100_low600_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test22

# 4d experiments with low of -1000, 20 adv
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 20 --low_reward -1000 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 20 --exp_title linear_20adv_d4_conc100_h100_low1000_r2 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=test23

# domain randomization in 2d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test24

# domain randomization in 4d
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 30000 --num_cpus 8 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc100_h100_r1 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --num_concat_states 100 --horizon 100 --scaling -0.8 --adv_strength 0.1 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test25
