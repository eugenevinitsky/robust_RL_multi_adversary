# D-MALT: Diverse Multi-Adversarial Learning for Transfer
We investigate the effect of sim2real via multiple types of parametrized adversaries

## Setup instructions
To install the code with anaconda run 
- `conda env create -f environment.yml`
- `conda activate sim2real`
- `python setup.py develop` 
- Then go follow the install instructions at: https://github.com/sybrenstuvel/Python-RVO2 (TBD if this is necessary)
- If you want to run videos make sure to install ffmpeg. If you have brew you can run `brew install ffmpeg`.
- There is a ray error so once you have set everything up run `python utils.py replace_rnn_sequence.py`

## Running experiments on EC2
1. Get AWS account login/credentials from Aboudy / Eugene
- Create an S3 bucket i.e. s3://yourname.experiments
- pip install awscli
    - Run aws configure
    - Use credentials provided by eugene / aboudy
    - region: `us-west-1`
    - default output format: `json`
- Create a new branch for experiments
- Modify `ray_autoscale.yaml`
    - Head & Worker Instance Type: m4.16xlarge or c4.4xlarge
    - Set init_workers=min_workers=max_workers to the number of parallel nodes that you want
    - Change `master` to your branch name`

Executing the run script as follows:

`ray exec autoscale.yaml "python ~/adversarial_sim2real/run_scripts/test_rllib_script.py --use_s3 --multi_node <exp_name>" --cluster-name <cluster_name> --start --stop --tmux`

- `use_s3` uploads experiment results to `s3://eugene.experiments/sim2real/`
- `multi_node` enables parallelization across instances in the cluster

## Unit tests
To run unit tests cd to the outer directory and run `python -m pytest`. Please do this before
merging any PR. This will be automated later.