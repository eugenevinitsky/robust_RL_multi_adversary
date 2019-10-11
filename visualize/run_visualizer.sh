#!/usr/bin/env bash

# Example: "python rollout.py <PATH_TO_POLICY> <CHECKPOINT_NUM>
# For videos add "--video_file <OUTPUT_PATH/FILENAME"
# To just display the video but not save it add --traj
python rollout.py /Users/eugenevinitsky/ray_results/sim2_images/PPO_CrowdSim_0_2019-10-08_10-16-15qo7o16tq 1 --video_file ~/test --show_images