#!/usr/bin/env bash

# 11/25 experiments

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py MA_0diff_1ad --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1" --start --stop --cluster-name=ma_test

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py MA_0diff_5ad --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5" --start --stop --cluster-name=ma_test