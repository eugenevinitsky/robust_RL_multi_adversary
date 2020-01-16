#!/usr/bin/env bash

## 1/8/2019

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p2str --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3 --use_lstm" --start --stop --tmux --cluster-name=ev_pend_test
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_nolstm_test_0p2str --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3" --start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p1str --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --use_lstm" --start --stop --tmux --cluster-name=ev_pend_test3
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_nolstm_test_012str --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3" --start --stop --tmux --cluster-name=ev_pend_test4
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_nolstm_test_noadv --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --run_transfer_tests\
# --grid_search --use_s3 --num_adv 0" --start --stop --tmux --cluster-name=ev_pend_test5

############################################################################################################################################################
## 1/9/2019
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p2str_conc4 --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3 --num_concat_states 4" --start --stop --tmux --cluster-name=ev_pend_test1
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p4str_conc4 --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.4 --run_transfer_tests --grid_search --use_s3 --num_concat_states 4" --start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p6str_conc4 --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 4" --start --stop --tmux --cluster-name=ev_pend_test3
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p2str_conc10 --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test4
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p4str_conc10 --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.4 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test5
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p6str_conc10 --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test6

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p2str_conc4_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3 --num_concat_states 4" --start --stop --tmux --cluster-name=ev_pend_test7
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p4str_conc4_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.4 --run_transfer_tests --grid_search --use_s3 --num_concat_states 4" --start --stop --tmux --cluster-name=ev_pend_test8
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p6str_conc4_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 4" --start --stop --tmux --cluster-name=ev_pend_test9
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p2str_conc10_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test10
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p4str_conc10_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.4 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test11
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_lstm_test_0p6str_conc10_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test12

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_test_0p6str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test13
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_test_0p6str_conc10_600itr_randstatefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test14

#############################################################################################################################################################
# 01/10 exps
#############################################################################################################################################################
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p6str_conc10_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test1
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p6str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p6str_conc10_600itr_randstatefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test3
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p3str_conc10_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test4
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p3str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test5
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p3str_conc10_600itr_randstatefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test6
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p1str_conc10_600itr --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test7
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p1str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test8
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p1str_conc10_600itr_randstatefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test9

#############################################################################################################################################################
# 01/11 exps
#############################################################################################################################################################

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p1str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test1
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p3str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p6str_conc10_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test3

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p1str_conc30_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 30" --start --stop --tmux --cluster-name=ev_pend_test4
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p3str_conc30_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 30" --start --stop --tmux --cluster-name=ev_pend_test5
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p6str_conc30_600itr_statefunc --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 30" --start --stop --tmux --cluster-name=ev_pend_test6
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p1str_conc30_600itr_statefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 30" --start --stop --tmux --cluster-name=ev_pend_test7
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p3str_conc30_600itr_statefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 30" --start --stop --tmux --cluster-name=ev_pend_test8
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pendulum_0p6str_conc30_600itr_statefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.6 --run_transfer_tests --grid_search --use_s3 --num_concat_states 30" --start --stop --tmux --cluster-name=ev_pend_test9

#############################################################################################################################################################
# 01/13 exps
#############################################################################################################################################################
## COS Experiments
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_cos_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test1
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_cos_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_cos_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test3
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_cos_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test4
#
## STATE FUNC Experiments
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_statefunc_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test5
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_statefunc_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test6
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_statefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test7
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_statefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test8
#
## RAND STATE FUNC Experiments
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_randstatefunc_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test9
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_randstatefunc_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test10
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_randstatefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test11
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_randstatefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test12
#
## Friction FUNC Experiments
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_fric_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test13
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_fric_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test14
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_fric_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test15
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_fric_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test16
#
## RAND Friction FUNC Experiments
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_randfric_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test17
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_randfric_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_friction \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test18
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc10_randfric_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test19
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc10_randfric_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_friction \
#--guess_next_state --adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test20
#
## 1-step memory experiments
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_cos_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test21
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_statefunc_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test22
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randstatefunc_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test23
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test24
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randfriction_5adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test25
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_cos_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type cos \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test26
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_statefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test27
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randstatefunc_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_state_func \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test28
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test29
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randfriction_15adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test30
#
## The base case. We better hope everything outperforms this
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_0adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 0 --model_based --adversary_type rand_friction \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test31
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_1p0str_conc1_0adv --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 0 --model_based --adversary_type rand_friction \
#--adv_strength 1.0 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test32

# The LSTM experiments

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_cos_5adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type cos --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test21
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_statefunc_5adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test22
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randstatefunc_5adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_state_func --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test23
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_5adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type friction --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test24
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randfriction_5adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type rand_friction --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test25
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_cos_15adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type cos --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test26
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_statefunc_15adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type state_func --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test27
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randstatefunc_15adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_state_func --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test28
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_15adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type friction --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test29
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_randfriction_15adv_lstm --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 600 --num_cpus 4 --num_adv 15 --model_based --guess_adv --adversary_type rand_friction --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1" --start --stop --tmux --cluster-name=ev_pend_test30


#############################################################################################################################################################
# 01/13 exps
# Experiments with a bigger model to understand why the 1 concat is winning
#############################################################################################################################################################

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_5adv_lstm_len1000 --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --guess_adv --adversary_type friction --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test1
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_15adv_lstm_len1000 --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 15 --model_based --guess_adv --adversary_type friction --use_lstm \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test2
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_conc1_friction_5adv_len1000 --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --guess_adv --adversary_type friction \
#--guess_next_state --adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test3
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_lstm_len1000_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test4
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_15adv_lstm_len1000_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 15 --model_based --adversary_type friction --use_lstm \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test5
#
## baseline with an LSTM
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_lstm_0adv_len1000_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 0 --run_transfer_tests --grid_search --use_s3 \
#--num_concat_states 1 --horizon 1000 --use_lstm" --start --stop --tmux --cluster-name=ev_pend_test6
#
## These are some baselines without an LSTM
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_len1000_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test7
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_0adv_len1000_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 0 --run_transfer_tests --grid_search --use_s3 \
#--num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test8
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_15adv_len1000_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 15 --model_based --adversary_type friction \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 1000" --start --stop --tmux --cluster-name=ev_pend_test9

### 500 horizon experiments #####################################################################################

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_lstm_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test10
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_15adv_lstm_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 15 --model_based --adversary_type friction --use_lstm \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test11
#
## baseline with an LSTM
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_lstm_0adv_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 0 --run_transfer_tests --grid_search --use_s3 \
#--num_concat_states 1 --horizon 500 --use_lstm" --start --stop --tmux --cluster-name=ev_pend_test12
#
## These are some baselines without an LSTM
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --num_samples 2 \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test13
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_0adv_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 0 --run_transfer_tests --grid_search --use_s3 --num_samples 2 \
#--num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test14
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_15adv_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 15 --model_based --adversary_type friction --num_samples 2 \
#--adv_strength 0.5 --run_transfer_tests --grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test15

# Just trying to do some hyper parameter tuning for the pendulum

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_lstm_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test16
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_conc20_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 20 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test17

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_base_lstm_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test18
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_0adv_base_lstm_len500_NG --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 500 --num_cpus 3 --num_adv 0 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test19

#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_base_lstm_len500_NG_1000iter --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test20
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_base_lstm_len500_NG_1000iter_vf100 --train_batch_size 10000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test21
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_base_lstm_len500_NG_1000iter_tb20000 --train_batch_size 20000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test22
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_base_lstm_len500_guessadv_1000iter_tb20000 --train_batch_size 20000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node --guess_adv \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use\
#_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test23
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title pend_0p5str_friction_5adv_concat20_len500_1000iter_tb20000 --train_batch_size 20000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 20 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test24
#
#ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
#--exp_title high_lr_friction_5adv_base_lstm_len500_NG_1000iter_tb20000 --train_batch_size 20000 \
#--checkpoint_freq 50 --num_iters 1000 --num_cpus 3 --num_adv 5 --model_based --adversary_type friction --use_lstm --multi_node \
#--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test25

#############################################################################################################################################################
# 01/16 exps
# Experiments with a bigger model to understand why the 1 concat is winning
#############################################################################################################################################################
ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title big_grid_friction_5adv_conc1_len500 --train_batch_size 20000 \
--checkpoint_freq 50 --num_iters 1000 --num_cpus 6 --num_adv 5 --model_based --adversary_type friction --multi_node \
--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 1 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test1

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title big_grid_friction_5adv_conc10_len500 --train_batch_size 20000 \
--checkpoint_freq 50 --num_iters 1000 --num_cpus 6 --num_adv 5 --model_based --adversary_type friction --multi_node \
--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 10 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test2

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title big_grid_friction_5adv_conc20_len500 --train_batch_size 20000 \
--checkpoint_freq 50 --num_iters 1000 --num_cpus 6 --num_adv 5 --model_based --adversary_type friction --multi_node \
--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 10 --horizon 500" --start --stop --tmux --cluster-name=ev_pend_test3

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title big_grid_friction_5adv_conc20_len500_lstm_3iters --train_batch_size 20000 \
--checkpoint_freq 50 --num_iters 1000 --num_cpus 6 --num_adv 5 --model_based --adversary_type friction --multi_node \
--adv_strength 0.5 --run_transfer_tests --big_grid_search --use_s3 --num_concat_states 10 --horizon 500 --num_sgd_iters 3" --start --stop --tmux --cluster-name=ev_pend_test4
