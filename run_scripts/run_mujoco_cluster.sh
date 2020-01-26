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
## 1/23/20 tests. Tests for hopper

## No adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
#--num_adv_strengths 0 --advs_per_strength 0 --adv_strength 10.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_0adv_0ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test1
#
## 1 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 1 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_1ps_str10p0_concat1_curr --num_cpus 2 --run_transfer_tests --goal_score 3000 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test2
#
## 1 adversary no curriculum
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 1 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_1ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test9
#
## We test the hypothesis that lots of adversaries in the same strength level is good. They may pick different strategies if they all have
## the same strength
## curriculum with 2 adversaries per strength and no memory and low goal
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test3
#
## curriculum with 5 adversaries per strength and no memory and low goal
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
#--num_adv_strengths 2 --advs_per_strength 5 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_2adv_5ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test4
#
## Just testing the no curricula case. Same logic in adversary number as above.
## curriculum with 2 adversaries per strength and no memory and no curriculum
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
#--num_adv_strengths 5 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_5adv_2ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test5
#
## curriculum with 5 adversaries per strength and no memory and no curriculum
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
#--num_adv_strengths 2 --advs_per_strength 5 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_2adv_5ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test6
#
### These test the hypothesis that because of the small batch size, too many adversaries leads to the adversaries not training
## curriculum with 2 adversaries per strength, 4 adv total and no memory and no curriculum
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --concat_actions --num_concat_states 1 \
#--num_adv_strengths 2 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_2adv_2ps_str5p0_concat1 --num_cpus 2 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test7
#
## curriculum with 2 adversaries per strength, 4 total, and no memory and low goal
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 10000 --num_iters 1000 --checkpoint_freq 100 --curriculum --concat_actions --num_concat_states 1 \
#--num_adv_strengths 2 --advs_per_strength 2 --adv_strength 5.0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_2adv_5ps_str5p0_concat1_curr_gs2500 --num_cpus 2 --run_transfer_tests --goal_score 2500 --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test8

######################################################################################################################################
## 1/23/20 tests. Tests for half cheetah

## HOPPER EXPS
## No adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 0 --advs_per_strength 0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_0adv_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test1
#
## 1 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 1 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_1ps_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 10 \
#--num_adv_strengths 1 --advs_per_strength 1 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_1ps_concat10_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test13
#
## 3 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 3 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_3adv_1ps_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 10 \
#--num_adv_strengths 1 --advs_per_strength 3 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_3adv_1ps_concat10_bigbatch --num_cpus 10 --run_transfer_tests --multi_node --concat_actions" \
#--start --stop --tmux --cluster-name=ev_pend_test4
#
## 5 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_5ps_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 10 \
#--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_1adv_5ps_concat10_bigbatch --num_cpus 10 --run_transfer_tests --multi_node --concat_actions" \
#--start --stop --tmux --cluster-name=ev_pend_test6
#
## CHEETAH EXPS
## No adversaries
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 0 --advs_per_strength 0 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_0adv_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test7
#
## 1 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 1 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_1adv_1ps_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test8
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 10 \
#--num_adv_strengths 1 --advs_per_strength 1 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_1adv_1ps_concat10_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test14
#
## 3 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 3 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_3adv_1ps_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test9
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 10 \
#--num_adv_strengths 1 --advs_per_strength 3 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_3adv_1ps_concat10_bigbatch --num_cpus 10 --run_transfer_tests --multi_node --concat_actions" \
#--start --stop --tmux --cluster-name=ev_pend_test10
#
## 5 adversary
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_1adv_5ps_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
#--start --stop --tmux --cluster-name=ev_pend_test11
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 10 \
#--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name cheetah \
#--exp_title cheetah_1adv_5ps_concat10_bigbatch --num_cpus 10 --run_transfer_tests --multi_node --concat_actions" \
#--start --stop --tmux --cluster-name=ev_pend_test12
#
## domain randomization
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 0 --advs_per_strength 0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_0adv_concat1_bigbatch_dr_cheat --num_cpus 10 --run_transfer_tests --multi_node \
#--domain_randomization --cheating" \
#--start --stop --tmux --cluster-name=ev_pend_test15
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
#--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
#--num_adv_strengths 0 --advs_per_strength 0 --grid_search --use_s3 --env_name hopper \
#--exp_title hop_0adv_concat1_bigbatch_dr --num_cpus 10 --run_transfer_tests --multi_node --domain_randomization" \
#--start --stop --tmux --cluster-name=ev_pend_test16


######################################################################################################################################
## 1/26/20 tests. Tests for diverse hopper

## No adversaries
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 \
--num_adv_strengths 0 --advs_per_strength 0 --grid_search --use_s3 --env_name hopper \
--exp_title hop_0adv_concat1_bigbatch --num_cpus 10 --run_transfer_tests --multi_node" \
--start --stop --tmux --cluster-name=ev_pend_test1

## 5 adversaries, goal scores of 4000 and 0
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_concat1_rew_l0_h4000 --num_cpus 10 --run_transfer_tests --multi_node --reward_range \
--low_reward 0 --high_reward 4000" \
--start --stop --tmux --cluster-name=ev_pend_test2

## 5 adversaries, goal scores of 4500 and 2000
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_concat1_rew_l3000_h4000 --num_cpus 10 --run_transfer_tests --multi_node --reward_range \
--low_reward 2000 --high_reward 4500" \
--start --stop --tmux --cluster-name=ev_pend_test3

## 5 adversaries, goal scores of 4500 and 3500
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_concat1_rew_l3500_h4000 --num_cpus 10 --run_transfer_tests --multi_node --reward_range \
--low_reward 3500 --high_reward 4500" \
--start --stop --tmux --cluster-name=ev_pend_test4

## 5 adversaries, just the l2 difference reward with coeff 0.5
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_concat1_l2rew0p5 --num_cpus 10 --run_transfer_tests --multi_node --l2_reward --l2_reward_coeff 0.5" \
--start --stop --tmux --cluster-name=ev_pend_test5

## 5 adversaries, just the l2 difference reward with coeff 5.0
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_concat1_l2rew5p0 --num_cpus 10 --run_transfer_tests --multi_node --l2_reward --l2_reward_coeff 5.0" \
--start --stop --tmux --cluster-name=ev_pend_test6

## 5 both l2 reward and reward goals
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 1 --advs_per_strength 5 --grid_search --use_s3 --env_name hopper \
--exp_title hop_5adv_concat1_rew_l3000_h4000_l2rew0p5 --num_cpus 10 --run_transfer_tests --multi_node --reward_range \
--low_reward 2000 --high_reward 4500 --l2_reward" \
--start --stop --tmux --cluster-name=ev_pend_test7

## 9 with both l2 rew and reward goals but with comparisons only in tranche
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_adv_lerrel.py \
--train_batch_size 100000 --num_iters 1000 --checkpoint_freq 100 --num_concat_states 1 --concat_actions \
--num_adv_strengths 3 --advs_per_strength 3 --grid_search --use_s3 --env_name hopper \
--exp_title hop_6adv_concat1_rew_l3000_h4000_l2rew0p5_tranche --num_cpus 10 --run_transfer_tests --multi_node --reward_range \
--low_reward 2000 --high_reward 4500 --l2_reward --l2_in_tranche" \
--start --stop --tmux --cluster-name=ev_pend_test8