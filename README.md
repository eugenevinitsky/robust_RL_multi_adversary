# D-MALT: Diverse Multi-Adversarial Learning for Transfer
We investigate the effect of sim2real via multiple types of parametrized adversaries

## Setup instructions
To install the code with anaconda run 
- `conda env create -f environment.yml`
- `conda activate sim2real`
- `python setup.py develop` 
- If you want to run videos make sure to install ffmpeg. If you have brew you can run `brew install ffmpeg`.

## Results reproduction
To simply rerun the plot generation on existing data just run:
`python visualize/generate_all_plots.py`

To recreate the results completely go to `run_scripts/recreate_results` and read 
and run the files there. However, to do this you will either need an AWS account that is properly
set up or to run each python command in there locally.

## Unit tests
To run unit tests cd to the outer directory and run `python -m pytest`. Please do this before
merging any PR. This will be automated later.