# 0 adv grid search
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name finger --exp_title finger_0adv_concat1_grid --num_cpus 10 --run_transfer_tests --multi_node --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_finger_test1

# 1 adv grid search
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name finger --exp_title finger_1adv_concat1_grid_str0p15 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions --clip_actions" --start --stop --tmux --cluster-name=yd_finger_test2

# 3 adv grid search w/ no reward ranges
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name finger --exp_title finger_3adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions --clip_actions" --start --stop --tmux --cluster-name=yd_finger_test6


# 5 adv grid search w/ no reward ranges
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name finger --exp_title finger_5adv_concat1_grid_str0p15_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.15 --grid_search --adv_all_actions --concat_actions --clip_actions" --start --stop --tmux --cluster-name=yd_finger_test3


# Domain randomization experiment w/ grid search
ray exec autoscale_yd.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 200 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name finger --exp_title finger_0adv_concat1_grid_dr --num_cpus 10 --run_transfer_tests --multi_node --grid_search --domain_randomization --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=yd_finger_test9

