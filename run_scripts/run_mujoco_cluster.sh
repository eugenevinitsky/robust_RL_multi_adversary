#!/usr/bin/env bash

######################################################################################################################################
## 1/22/20 tests. Tests for hopper

# curriculum with 2 adversaries per strength and memory
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 20 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str5p0_concat20_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test1

# curriculum with 2 adversaries per high strength and memory
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 20 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str10p0_concat20_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test2
#
## curriculum with 2 adversaries per strength and no memory
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --num_concat_states 1 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str5p0_concat1_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test3
#
## curriculum with 2 adversaries per high strength and no memory
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --num_concat_states 1 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test4

# 1 adversary and no memory
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 1 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_1ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test5
#
## 1 adversary and memory
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 20 \
#--num_adv_strengths 1 --advs_per_strength 1 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_1ps_str10p0_concat20_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test6

# No adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
#--num_adv_strengths 0 --advs_per_strength 0 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_0adv_0ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test7
#
## curriculum with 2 adversaries per strength and memory and low goal
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 20 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str5p0_concat20_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test8
#
## curriculum with 2 adversaries per strength and no memory and low goal
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test9

######################################################################################################################################
## 1/22/20 tests. Tests for hopper

# No adversaries
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
--num_adv_strengths 0 --advs_per_strength 0 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_0adv_0ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test1

# 1 adversary
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --num_concat_states 1 \
--num_adv_strengths 1 --advs_per_strength 1 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_1adv_1ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test2

# 1 adversary no curriculum
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
--num_adv_strengths 1 --advs_per_strength 1 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_1adv_1ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test9

# We test the hypothesis that lots of adversaries in the same strength level is good. They may pick different strategies if they all have
# the same strength
# curriculum with 2 adversaries per strength and no memory and low goal
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_2ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test3

# curriculum with 5 adversaries per strength and no memory and low goal
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
--num_adv_strengths 2 --advs_per_strength 5 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_2adv_5ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test4

# Just testing the no curricula case. Same logic in adversary number as above.
# curriculum with 2 adversaries per strength and no memory and no curriculum
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_2ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test5

# curriculum with 5 adversaries per strength and no memory and no curriculum
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
--num_adv_strengths 2 --advs_per_strength 5 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_2adv_5ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test6

## These test the hypothesis that because of the small batch size, too many adversaries leads to the adversaries not training
# curriculum with 2 adversaries per strength, 4 adv total and no memory and no curriculum
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
--num_adv_strengths 2 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_2adv_2ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test7

# curriculum with 2 adversaries per strength, 4 total, and no memory and low goal
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
--num_adv_strengths 2 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_2adv_5ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test8
