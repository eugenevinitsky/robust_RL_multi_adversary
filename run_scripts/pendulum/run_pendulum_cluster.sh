#!/usr/bin/env bash

## 1/8/2019

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_lstm_test_0p2str --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3 --use_lstm" --start --stop --tmux --cluster-name=ev_pend_test

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_nolstm_test_0p2str --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
--guess_next_state --adv_strength 0.2 --run_transfer_tests --grid_search --use_s3" --start --stop --tmux --cluster-name=ev_pend_test2

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_lstm_test_0p1str --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3 --use_lstm" --start --stop --tmux --cluster-name=ev_pend_test3

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_nolstm_test_012str --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --num_adv 5 --model_based --guess_adv \
--guess_next_state --adv_strength 0.1 --run_transfer_tests --grid_search --use_s3" --start --stop --tmux --cluster-name=ev_pend_test4

ray exec ././../../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/pendulum/run_pendulum.py \
--exp_title pendulum_nolstm_test_noadv --train_batch_size 10000 \
--checkpoint_freq 20 --num_iters 300 --num_cpus 4 --run_transfer_tests\
 --grid_search --use_s3 --num_adv 0" --start --stop --tmux --cluster-name=ev_pend_test5