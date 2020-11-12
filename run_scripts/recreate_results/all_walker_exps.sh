# 11/11/2020

# 0 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 0 --advs_per_strength 0 --advs_per_rew 0 --num_adv_rews 0 --use_s3 --env_name walker --exp_title walk_0adv_concat1_grid --num_cpus 10 --run_transfer_tests --multi_node --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=ev_walk_test1

# 1 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 1 --advs_per_rew 1 --num_adv_rews 1 --use_s3 --env_name walker --exp_title walk_1adv_concat1_grid_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=ev_walk_test2

# 3 adv grid search
ray exec autoscale.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 3 --advs_per_rew 1 --num_adv_rews 3 --use_s3 --env_name walker --exp_title walk_3adv_concat1_grid_str0p25 --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=ev_walk_test3

# 5 adv grid search w/ no reward ranges
ray exec autoscale.yaml "python /home/ubuntu/robust_RL_multi_adversary/run_scripts/mujoco/run_adv_mujoco.py --train_batch_size 100000 --num_iters 700 --checkpoint_freq 100 --num_concat_states 1 --num_adv_strengths 1 --advs_per_strength 5 --advs_per_rew 1 --num_adv_rews 5 --use_s3 --env_name walker --exp_title walk_5adv_concat1_grid_str0p25_norew --num_cpus 10 --run_transfer_tests --multi_node --adv_strength 0.25 --grid_search --adv_all_actions --concat_actions" --start --stop --tmux --cluster-name=ev_walk_test4
