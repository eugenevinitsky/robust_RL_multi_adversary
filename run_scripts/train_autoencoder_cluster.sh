#!/usr/bin/env bash

ray exec ../autoscale.yaml "python train_autoencoder.py --img_freq 20 --kernel_size 2 --total_step_num 400 --num_iters 40 --use_s3" \
--start --stop --cluster-name=ev_auto