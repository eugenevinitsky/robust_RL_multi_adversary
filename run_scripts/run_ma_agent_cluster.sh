#!/usr/bin/env bash

# 11/25 experiments

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 30" \
#--tmux --start --stop --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 30" \
#--tmux --start --stop --cluster-name=ma_test2

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 30" \
--tmux --start --stop --cluster-name=ma_test3

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PS --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5 --perturb_state --num_cpus 30" \
--tmux --start --stop --cluster-name=ma_test4

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 30" \
--tmux --start --stop --cluster-name=ma_test5

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PS_PA --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5 --perturb_state --perturb_actions --num_cpus 30" \
--tmux --start --stop --cluster-name=ma_test6