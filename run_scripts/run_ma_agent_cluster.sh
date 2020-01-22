#!/usr/bin/env bash

# 11/25 experiments

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PS --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5 --perturb_state --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PS_PA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 5 --perturb_state --perturb_actions --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test6

#######################################################################################################################################
### 11/26 experiments, rerunning the exps but the adversary has a much smaller action space now
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=ma_test6

######################################################################################################################################
## 12/1/19 experiments, rerunning the exps but with 4 humans
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h4 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA_h4 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_h4 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_h4 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA_h4 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA_h4 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=ma_test6

#######################################################################################################################################
### 12/2/19 experiments, rerunning the exps but with 4 humans
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test6

#######################################################################################################################################
### 12/3/19 experiments, rerunning the exps but with 4 humans
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test6

#######################################################################################################################################
### 12/4/19 experiments, rerunning the exps but with 3 humans
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 400 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test6


#######################################################################################################################################
### 12/4/19 experiments, rerunning the exps but with 3 humans
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 500 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test6

######################################################################################################################################
## 12/6/19 experiments, rerunning the exps but with a way longer pre-training time
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 3 --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 1 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 3 --perturb_state --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 1 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_3ad_PS_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 3 --perturb_state --perturb_actions --num_cpus 4 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ma_test6

######################################################################################################################################
## 12/17/19 tests
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 3 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 3 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_10ad_PA_h3 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 3 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_1ad_PA_h1 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 1 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test4
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_5ad_PA_h1 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 1 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0diff_10ad_PA_h1 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 1 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test6

######################################################################################################################################
## 12/26/19 tests
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0ad_PA_h0_pred_freq1_sc5 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 0 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --cluster-name=ev_ma_test9
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_1ad_PA_h0_pred_freq1_sc5 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 1 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --cluster-name=ev_ma_test10
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_5ad_PA_h0_pred_freq1_sc5 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --cluster-name=ev_ma_test11
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_PA_h0_pred_freq1_sc5 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --cluster-name=ev_ma_test12

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0ad_PA_h0_GA --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 0 --perturb_actions --add_gaussian_noise_action --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --cluster-name=ev_ma_test13


######################################################################################################################################
## 12/27/19 tests
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0ad_PA_h0 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 1600 --kl_diff_weight 0 --num_adv 0 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_5ad_PA_h0_pred_freq1_sc5 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 1600 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0ad_PA_h0_GA --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 1600 --kl_diff_weight 0 --num_adv 0 --perturb_actions --add_gaussian_noise_action --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_PA_h0_pred_freq1_sc5 --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 50 --num_iters 1600 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test4

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_5ad_PA_h0_pred_freq1_sc5_DDPG --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 500 --num_iters 4000 --kl_diff_weight 0 --num_adv 5 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search --algorithm DDPG" \
#--tmux --start --stop --cluster-name=ev_ma_test5
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_PA_h0_pred_freq1_sc5_DDPG --use_s3 --num_samples 1 \
#--train_batch_size 15000 --checkpoint_freq 500 --num_iters 4000 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search --algorithm DDPG" \
#--tmux --start --stop --cluster-name=ev_ma_test6

######################################################################################################################################
## 12/30/19 tests

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_0ad_PA_h0 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 0 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test1
##
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_PA_h0_pred_freq1 --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_PA_h0_pred_freq1_switch --use_s3 --num_samples 1 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
#--tmux --start --stop --cluster-name=ev_ma_test3

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_kd_p001_PA_h0_pred_freq1_switch --use_s3 --num_samples 1 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0.001 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
--tmux --start --stop --cluster-name=ev_ma_test4

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_kd_p01__PA_h0_pred_freq1_switch --use_s3 --num_samples 1 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0.01 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
--tmux --start --stop --cluster-name=ev_ma_test5

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_kd_p1__PA_h0_pred_freq1_switch --use_s3 --num_samples 1 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0.1 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
--tmux --start --stop --cluster-name=ev_ma_test6

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/ma_crowd.py --exp_title MA_10ad_kd_0p0__PA_h0_pred_freq1_switch --use_s3 --num_samples 1 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 800 --kl_diff_weight 0.0 --num_adv 10 --perturb_actions --num_cpus 8 --run_transfer_tests --human_num 0 --grid_search" \
--tmux --start --stop --cluster-name=ev_ma_test7

