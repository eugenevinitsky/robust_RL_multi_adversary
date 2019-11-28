#!/usr/bin/env bash

# Example: "python rollout.py <PATH_TO_POLICY> <CHECKPOINT_NUM>
# For videos add "--video_file <OUTPUT_PATH/FILENAME"
# To just display the video but not save it add --traj
python transfer_test.py /Users/eugenevinitsky/Desktop/Research/Data/sim2real/11-26-2019/SA_GaA/SA_GaA/PPO_CrowdSim_0_2019-11-26_18-07-09i95vfcj9 \
500 --num_rollouts 40