#!/usr/bin/env bash

ray exec autoscale.yaml "python adversarial_sim2real/test_rllib_script.py --exp_title test_memory --num_cpus 32 --num_iters 100" \
    --start --stop --cluster-name=test_memory