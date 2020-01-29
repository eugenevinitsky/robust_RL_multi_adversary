#!/usr/bin/env bash

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 5 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2_conc20_h20 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 20 --horizon 20" \
#--start --stop --tmux --cluster-name=ev_lin_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 3 \
#--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 3 --exp_title linear_3adv_d2_conc20_h20 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node --num_concat_states 20 --horizon 20" \
#--start --stop --tmux --cluster-name=ev_lin_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
#--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 0 \
#--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2_conc20_h20 --run_transfer_tests \
#--grid_search --use_s3 --dim 2 --multi_node20 --num_concat_states 20 --horizon 20" \
#--start --stop --tmux --cluster-name=ev_lin_test3

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4_conc20_h20 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -1.0 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test4

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 3 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 3 --exp_title linear_3adv_d4_conc20_h20 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -1.0 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test5

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4_conc20_h20 --run_transfer_tests \
--grid_search --use_s3 --dim 4 --multi_node --agent_strength 0.4 --scaling -1.0 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test6

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_3adv_d8_conc20_h20 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -2.2 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test7

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 3 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 3 --exp_title linear_5adv_d8_conc20_h20 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -2.2 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test8

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/linear_env/run_linear_env.py \
--num_iters 500 --train_batch_size 20000 --num_cpus 6 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8_conc20_h20 --run_transfer_tests \
--grid_search --use_s3 --dim 8 --multi_node --agent_strength 0.4 --scaling -2.2 --num_concat_states 20 --horizon 20" \
--start --stop --tmux --cluster-name=ev_lin_test9