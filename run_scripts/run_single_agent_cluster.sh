#!/usr/bin/env bash

# 11/25/19
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --num_cpus 30 --run_transfer_tests" --tmux --start --stop --cluster-name=sa_test
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_state --num_cpus 30 --run_transfer_tests" --tmux --start --stop --cluster-name=sa_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_action --num_cpus 30 --run_transfer_tests" --tmux --start --stop --cluster-name=sa_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_GaA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_state --add_gaussian_noise_action --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=sa_test4

####################################################################################################
# 11/26/19
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --num_cpus 30 --run_transfer_tests" --tmux --start --stop --cluster-name=sa_test
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_state --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=sa_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_action --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=sa_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_GaA --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_state --add_gaussian_noise_action --num_cpus 30 --run_transfer_tests" \
#--tmux --start --stop --cluster-name=sa_test4

####################################################################################################
# 12/1/19 running the exps with 4 humans to attempt to encourage it to avoid humans
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h4 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --num_cpus 18 --run_transfer_tests --human_num 4" --tmux --start --cluster-name=sa_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_h4 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_state --num_cpus 18 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=sa_test2

#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaA_h4 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_action --num_cpus 18 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=sa_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_GaA_h4 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 500 --add_gaussian_noise_state --add_gaussian_noise_action --num_cpus 18 --run_transfer_tests --human_num 4" \
#--tmux --start --cluster-name=sa_test4

#####################################################################################################
## 12/1/19 disable goal changing
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --num_cpus 34 --run_transfer_tests --human_num 3" --tmux --start --cluster-name=ev_sa_test1
#
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --add_gaussian_noise_action --num_cpus 34 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --add_gaussian_noise_state --num_cpus 34 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_GaA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --add_gaussian_noise_state --add_gaussian_noise_action --num_cpus 34 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test4

#####################################################################################################
## 12/3/19 disable goal changing
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --num_cpus 15 --run_transfer_tests --human_num 3" --tmux --start --cluster-name=ev_sa_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --add_gaussian_noise_action --num_cpus 15 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --add_gaussian_noise_state --num_cpus 15 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_GaA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 300 --add_gaussian_noise_state --add_gaussian_noise_action --num_cpus 15 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test4

#####################################################################################################
## 12/5/19 disable goal changing
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --num_cpus 9 --run_transfer_tests --human_num 3" --tmux --start --cluster-name=ev_sa_test1
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --add_gaussian_noise_action --num_cpus 9 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test2
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --add_gaussian_noise_state --num_cpus 9 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test3
#
#ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_GaS_GaA_h3 --use_s3 \
#--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --add_gaussian_noise_state --add_gaussian_noise_action --num_cpus 9 --run_transfer_tests --human_num 3" \
#--tmux --start --cluster-name=ev_sa_test4


####################################################################################################
# 12/8/19 disable goal changing
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h3_0hum_fixedgoal --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --num_cpus 9 --run_transfer_tests --human_num 0" --tmux --start --cluster-name=ev_sa_test1

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h3_3hum_fixedgoal --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --num_cpus 9 --run_transfer_tests --human_num 3" --tmux --start --cluster-name=ev_sa_test2

ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h3_3hum_unfixedgoal --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --num_cpus 9 --run_transfer_tests --human_num 3" --tmux --start --cluster-name=ev_sa_test3

####################################################################################################
# 12/11/19 add heading to state space, remove LSTM
ray exec ../autoscale.yaml "python /home/ubuntu/adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title SA_h0_noLSTM_heading --use_s3 \
--train_batch_size 30000 --checkpoint_freq 50 --num_iters 400 --num_cpus 9 --run_transfer_tests --human_num 0" --tmux --start --cluster-name=ev_sa_test1