# Robust Reinforcement Learning via adversary pools
We investigate the effect of population based training on the robustness of solutions to the robust MDP.

## Setup instructions
To install the code with anaconda run 
- `conda env create -f environment.yml`
- `conda activate adversary_pools`
- `python setup.py develop` 
- If you want to run videos make sure to install ffmpeg. If you have brew you can run `brew install ffmpeg`.

## Results reproduction
To simply rerun the plot generation on existing data just run:
`python visualize/generate_all_plots.py`

To recreate the results completely go to `run_scripts/recreate_results` and read 
and run the files there. However, to do this you will either need an AWS account that is properly
set up or to run each python command in there locally.
