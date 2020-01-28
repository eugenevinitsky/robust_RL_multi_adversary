#!/usr/bin/env bash

end="$(cut -d'/' -f2 <<<"$1")"
mkdir "~/s3_test"
cd s3_data
aws s3 sync s3://sim2real/adv_robust/$1/$end ./$1 --exclude="*" --include="*params*"
aws s3 sync s3://sim2real/adv_robust/$1/$end ./$1 --exclude="*" --include="*checkpoint_1000*"
