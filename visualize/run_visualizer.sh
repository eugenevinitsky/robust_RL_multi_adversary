#!/usr/bin/env bash

# Example: "python rollout.py <PATH_TO_POLICY> <CHECKPOINT_NUM>
# For videos add "--video_file <OUTPUT_PATH/FILENAME"
# To just display the video but not save it add --traj
python rollout.py /Users/eugenevinitsky/ray_results/test_sim/PPO_CrowdSim_0_2019-09-20_21-04-25sewdtna3 1 --video_file ~/test