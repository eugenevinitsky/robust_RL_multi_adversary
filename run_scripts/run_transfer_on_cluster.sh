#!/usr/bin/env bash

#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_l2rew0p5_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_1
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_l2rew2p0_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_2
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_l2rew5p0_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_3
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_rew_l3000_h4000_l2rew0p5_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_4
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_4adv_concat1_rew_l3200_h3600_l2rew0p5_tranche_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_5
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_9adv_concat1_rew_l3000_h4000_l2rew0p5_tranche_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_6
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat8_rew_l3300_h3700_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_7
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat8_rew_l3100_h3900_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_8
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_l2rew0p1_r2 1000 \
#01-27-2020" --start --stop --tmux --cluster-name=ev_results_9

# results for 1/29 memory free ablation

#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_rew_l1000_h3500_str0p25_all 700 \
#01-29-2020" --start --stop --tmux --cluster-name=ev_results_1

# results from final ICML run, 2/01

ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_0adv_concat1_grid 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_1

#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_0adv_concat10_grid 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_2

#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_1adv_concat1_grid_str0p25 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_3
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_1adv_concat10_grid_str0p25 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_4
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_grid_str0p25rew_l1000_h3500 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_5
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat10_grid_str0p25rew_l1000_h3500 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_6
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat1_grid_str0p25_norew 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_7
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_5adv_concat10_grid_str0p25_norew 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_8
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_10adv_concat1_grid_str0p25rew_l1000_h3500 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_9

ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_10adv_concat10_grid_str0p25rew_l1000_h3500 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_10
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_10adv_concat1_grid_str0p25_norew 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_11
#
ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_10adv_concat10_grid_str0p25_norew 700 \
02-01-2020" --start --stop --tmux --cluster-name=ev_results_12

#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_0adv_concat1_grid_dr 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_13
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_0adv_concat10_grid_dr 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_14
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_0adv_concat1_grid_extremedr 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_15
#
#ray exec ../autoscale.yaml "python ~/adversarial_sim2real/run_scripts/cluster_transfer_tests.py hop_0adv_concat10_grid_extremedr 700 \
#02-01-2020" --start --stop --tmux --cluster-name=ev_results_16