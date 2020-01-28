#!/usr/bin/env bash

ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_l2rew0p1_r2 1000 \
01-27-2020" --start --stop --cluster-name=ev_results_1