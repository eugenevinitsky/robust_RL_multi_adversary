"""If compiling the results on the machine failed, you can use this script locally"""
import argparse
import os

import ray

from visualize.transfer_test import run_transfer_tests
from utils.parsers import replay_parser
from utils.rllib_utils import get_config_from_path

parser = argparse.ArgumentParser()
parser.add_argument('--output_file_name', type=str, default='transfer_out',
                    help='The file name we use to save our results')
parser.add_argument('--output_dir', type=str, default='transfer_results',
                    help='')

parser = replay_parser(parser)
args = parser.parse_args()

ray.init()

output_path = os.path.abspath(args.result_dir).split('/')[-1]
if not os.path.exists('transfer_results/{}'.format(output_path)):
    os.makedirs('transfer_results/{}'.format(output_path))

for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser(args.result_dir)):
    if "checkpoint_{}".format(args.checkpoint_num) in dirpath:

        # grab the experiment name
        folder = os.path.dirname(dirpath)
        tune_name = folder.split("/")[-1]
        inner_folder_name = os.path.dirname(folder).split("/")[-1]
        results_path = 'transfer_results/{}/{}'.format(output_path, inner_folder_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        config, checkpoint_path = get_config_from_path(folder, args.checkpoint_num)
        run_transfer_tests(config, checkpoint_path, args.num_rollouts, inner_folder_name, results_path,
                           save_trajectory=False, show_images=False)