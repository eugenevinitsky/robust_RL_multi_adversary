## 0 adv grid search
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name cup --exp_title cup_0adv_concat1_grid --num_cpus 10 --run_transfer_tests --multi_node --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_cup_test1
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py cup_0adv_concat1_grid 700 11-12-2020" --start --stop --tmux --cluster-name=yd_cup_ttest1
#
#
## 1 adv grid search
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name cup --exp_title cup_1adv_concat1_grid_str0p15 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions --clip_actions" --start --stop --tmux --cluster-name=yd_cup_test2
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py ant_1adv_concat1_grid_str0p25 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest2
#
## 3 adv grid search w/ no reward ranges
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name cup --exp_title cup_3adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions --clip_actions" --start --stop --tmux --cluster-name=yd_cup_test6
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py ant_5adv_concat1_grid_str0p25_norew 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest3
#
#
## 5 adv grid search w/ no reward ranges
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name cup --exp_title cup_5adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions --clip_actions" --start --stop --tmux --cluster-name=yd_cup_test3
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py ant_5adv_concat1_grid_str0p25_norew 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest3
#
#
## Domain randomization experiment w/ grid search
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name cup --exp_title cup_0adv_concat1_grid_dr --num_cpus 10 --run_transfer_tests --multi_node --grid_search --domain_randomization --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_cup_test9
#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py ant_0adv_concat1_grid_dr 700 05-17-2020" --start --stop --tmux --cluster-name=yd_ant_ttest4

#
####AZURE
#ray exec autoscale_azure.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name cup --exp_title cup_0adv_concat1_grid --num_cpus 10 --run_transfer_tests --multi_node --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_cup_test1
#

# 0 adv seed search
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py \
--train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 \
--advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name cup --exp_title cup_0adv_concat1_grid \
--num_cpus 10 --run_transfer_tests --multi_node --adv_all_actions --concat_actions \
--seed_search --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_cup_test1

#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py cup_1adv_concat1_grid_str0p15 200 01-27-2021" --start --stop --tmux --cluster-name=yd_cup_ttest1


#
## 1 adv grid search
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py \
--train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 \
--advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name cup --exp_title \
cup_1adv_concat1_grid_str0p15 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 \
--adv_all_actions --concat_actions --clip_actions \
--seed_search --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_cup_test2

#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py cup_1adv_concat1_grid_str0p15 200 01-27-2021" --start --stop --tmux --cluster-name=yd_cup_ttest2


# 3 adv grid search w/ no reward ranges
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py \
--train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 \
--advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name cup \
--exp_title cup_3adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests \
--multi_node --adv_strength 0.15 --adv_all_actions --concat_actions --clip_actions \
--seed_search --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_cup_test6

#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py cup_3adv_concat1_grid_str0p15_norew 200 01-27-2021" --start --stop --tmux --cluster-name=yd_cup_ttest3


# 5 adv grid search w/ no reward ranges
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py \
--train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 \
--advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name cup \
--exp_title cup_5adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node \
--adv_strength 0.15 --adv_all_actions --concat_actions --clip_actions \
--seed_search --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_cup_test3

#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py cup_5adv_concat1_grid_str0p15_norew 200 01-27-2021" --start --stop --tmux --cluster-name=yd_cup_ttest4

# Domain randomization experiment w/ grid search
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py \
--train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 \
--advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name cup --exp_title cup_0adv_concat1_grid_dr \
--num_cpus 10 --run_transfer_tests --multi_node --domain_randomization --adv_all_actions \
--concat_actions --seed_search --lambda_val 0.9 --lr .0005" --start --stop --tmux --cluster-name=yd_cup_test9

#ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/visualize/mujoco/cluster_transfer_tests.py cup_0adv_concat1_grid_dr 200 01-27-2021" --start --stop --tmux --cluster-name=yd_cup_ttest5
