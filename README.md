# Robust Reinforcement Learning via adversary pools
We investigate the effect of population based training on the robustness of solutions to the robust MDP.

## Setup instructions
To install the code with anaconda run 
- `conda env create -f environment.yml`
- `conda activate adversary_pools`
- `python setup.py develop` 
- If you want to run videos make sure to install ffmpeg. If you have brew you can run `brew install ffmpeg`.

## Running the envs
The relevant file is run_adv_mujoco.py in `run_scripts/mujoco`. For configurations, check out the options on the argparser.

## Results reproduction
To simply rerun the plot generation on existing data just run:
`python visualize/final_results/generate_all_plots.py`

To recreate the results completely go to `run_scripts/recreate_results` and read 
and run the files there. However, to do this you will either need an AWS account that is properly
set up or to run each python command in there locally.

## Citations
The original paper corresponding to this code is at https://arxiv.org/abs/2008.01825
For citations please use 
@article{vinitsky2020robust,
  title={Robust Reinforcement Learning using Adversarial Populations},
  author={Vinitsky, Eugene and Du, Yuqing and Parvate, Kanaad and Jang, Kathy and Abbeel, Pieter and Bayen, Alexandre},
  journal={arXiv preprint arXiv:2008.01825},
  year={2020}
}
