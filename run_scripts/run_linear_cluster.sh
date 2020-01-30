#!/usr/bin/env bash

## 2D
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 20 --horizon 20 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test1

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d2_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 20 --horizon 20 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test2
#
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 20 --horizon 20 --scaling -0.8 --adv_strength 0.2 --agent_strength 0.4" \
--start --stop --tmux --cluster-name=ev_lin_test3

## 4D
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.1 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test4

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d4_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.1 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test5

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.1 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test6

## 6D
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d6_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 6 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.066 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test7

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d6_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 6 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.066 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test8

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d6_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 6 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.066 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test9

## 8D
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 1 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 1 --exp_title linear_1adv_d8_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.05 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test10

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 5 \
--num_adv_strengths 1 --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.05 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test11

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 4 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8_conc20_h20_r4 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -0.8 --adv_strength 0.05 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test12
