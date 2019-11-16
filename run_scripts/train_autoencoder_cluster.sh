#!/usr/bin/env bash

ray exec ../autoscale.yaml "python adversarial_sim2real/run_scripts/train_autoencoder.py --img_freq 100 --kernel_size 2 --total_step_num 10000 --num_iters 2000 --gather_images --num_cpus 6--use_s3" \
--start --cluster-name=ev_auto