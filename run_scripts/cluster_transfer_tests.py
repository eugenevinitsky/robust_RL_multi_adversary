"""Run transfer tests on the cluster and upload them to AWS"""

import argparse
import errno
import os
import subprocess

import ray

from visualize.transfer_test import run_transfer_tests
from visualize.visualize_adversaries import visualize_adversaries
from utils.rllib_utils import get_config_from_path

parser = argparse.ArgumentParser()
parser.add_argument('exp_title', type=str)
parser.add_argument('checkpoint_num', type=int)
parser.add_argument('date', type=str, help='A date in M-DD-YYYY format')


args = parser.parse_args()


date = args.date

ray.init()

output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results'), date), args.exp_title)
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
    if "checkpoint_{}".format(args.checkpoint_num) in dirpath and 'test' not in dirpath:
        # grab the experiment name
        folder = os.path.dirname(dirpath)
        tune_name = folder.split("/")[-1]
        outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.expanduser(os.path.join(outer_folder, "visualize/transfer_test.py"))
        config, checkpoint_path = get_config_from_path(folder, str(args.checkpoint_num))

        run_transfer_tests(config, checkpoint_path, 500, args.exp_title, output_path, save_trajectory=False)
        visualize_adversaries(config, checkpoint_path, 20, 500, output_path)
        p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                         "s3://sim2real/transfer_results/{}/{}/{}".format(date, args.exp_title,
                                                                                                          tune_name)).split(
            ' '))

        p1.wait()