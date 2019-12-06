#!/usr/bin/env bash

cd ~/ray_results && rm -rf * && cd .. && cd adversarial_sim2real && git stash && git pull origin kl_div && cd ..