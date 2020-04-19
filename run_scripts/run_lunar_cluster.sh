#!/bin/bash

# 5 ADV RARL lunar lander w/ 8 states
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/lunar_lander/run_lunar_lander.py \
--train_batch_size 30000 --num_iters 500 --checkpoint_freq 100 --num_concat_states 10 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --grid_search --use_s3 \
--exp_title lunar_RARL_5adv_concat10_rid --num_cpus 9 --run_transfer_tests --multi_node \
--adv_strength 0.25 --adv_all_actions" \
--start --stop --tmux --cluster-name=ev_lun_test1

# 5 ADV DMALT lunar lander w/ 8 states
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 500 --checkpoint_freq 100 --num_concat_states 10 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --grid_search --use_s3 \
--exp_title lunar_DMALT_5adv_concat10_grid --num_cpus 9 --run_transfer_tests --multi_node \
--adv_strength 0.25 --adv_all_actions --reward_range --low_reward -200 --high_reward 300" \
--start --stop --tmux --cluster-name=ev_lun_test2

# 5 ADV RARL lunar lander w/o memory
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 500 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --grid_search --use_s3 \
--exp_title lunar_RARL_5adv_concat1_grid --num_cpus 9 --run_transfer_tests --multi_node \
--adv_strength 0.25 --adv_all_actions" \
--start --stop --tmux --cluster-name=ev_lun_test3

# 5 ADV DMALT lunar lander w/ memory
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 500 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --grid_search --use_s3 \
--exp_title lunar_DMALT_5adv_concat1_grid --num_cpus 9 --run_transfer_tests --multi_node \
--adv_strength 0.25 --adv_all_actions --reward_range --low_reward -200 --high_reward 300" \
--start --stop --tmux --cluster-name=ev_lun_test4