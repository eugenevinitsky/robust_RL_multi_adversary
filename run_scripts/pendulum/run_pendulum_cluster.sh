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

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_0p1str_conc10_600itr_statefunc --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test1

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_0p3str_conc10_600itr_statefunc --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 600 --num_cpus 4 --num_adv 5 --model_based --guess_adv --adversary_type state_func \
--guess_next_state --adv_strength 0.3 --run_transfer_tests --grid_search --use_s3 --num_concat_states 10" --start --stop --tmux --cluster-name=ev_pend_test2
