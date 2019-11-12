#!/usr/bin/env bash

ray exec autoscale.yaml "python adversarial_sim2real/run_scripts/test_rllib_script.py --exp_title test_memory2 --num_cpus 32 --num_iters 300 --train_on_images --use_s3" \
    --start --stop --cluster-name=test_memory2 --tmux