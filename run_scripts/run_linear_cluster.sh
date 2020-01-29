#!/usr/bin/env bash

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d2 --run_transfer_tests \
--grid_search --use_s3 --dim 2" \
--start --stop --tmux --cluster-name=ev_lin_test1

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 3 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 3 --exp_title linear_5adv_d2 --run_transfer_tests \
--grid_search --use_s3 --dim 2" \
--start --stop --tmux --cluster-name=ev_lin_test2

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d2 --run_transfer_tests \
--grid_search --use_s3 --dim 2" \
--start --stop --tmux --cluster-name=ev_lin_test3

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d4 --run_transfer_tests \
--grid_search --use_s3 --dim 4" \
--start --stop --tmux --cluster-name=ev_lin_test4

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 3 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 3 --exp_title linear_5adv_d4 --run_transfer_tests \
--grid_search --use_s3 --dim 4" \
--start --stop --tmux --cluster-name=ev_lin_test5

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d4 --run_transfer_tests \
--grid_search --use_s3 --dim 4" \
--start --stop --tmux --cluster-name=ev_lin_test6

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 5 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 5 --exp_title linear_5adv_d8 --run_transfer_tests \
--grid_search --use_s3 --dim 8" \
--start --stop --tmux --cluster-name=ev_lin_test7

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 3 \
--num_adv_strengths 1 --reward_range --advs_per_rew 1 --num_adv_rews 3 --exp_title linear_5adv_d8 --run_transfer_tests \
--grid_search --use_s3 --dim 8" \
--start --stop --tmux --cluster-name=ev_lin_test8

ray exec ../autoscale.yaml "python run_linear_env.py --num_iters 500 --train_batch_size 20000 --num_cpus 3 --advs_per_strength 0 \
--num_adv_strengths 0 --advs_per_rew 0 --num_adv_rews 0 --exp_title linear_dr_d8 --run_transfer_tests \
--grid_search --use_s3 --dim 8" \
--start --stop --tmux --cluster-name=ev_lin_test9