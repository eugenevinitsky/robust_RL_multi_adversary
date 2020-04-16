#!/usr/bin/env bash

mkdir "/home/ubuntu/s3_test"
cd "/home/ubuntu/s3_test"
aws s3 sync s3://sim2real/adv_robust/$1 ./ --exclude="*" --include="*params*"
aws s3 sync s3://sim2real/adv_robust/$1 ./ --exclude="*" --include="*checkpoint-*"