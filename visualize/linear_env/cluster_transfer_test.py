import argparse
import os
import ray
import errno
import subprocess
from utils.rllib_utils import get_config_from_path
from visualize.linear_env.visualize_adversaries import visualize_adversaries
from visualize.linear_env.transfer_test import run_transfer_tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_title', type=str)
    parser.add_argument('checkpoint_num', type=int)
    parser.add_argument('date', type=str, help='A date in M-DD-YYYY format')

    args = parser.parse_args()

    date = args.date

    ray.init()

    output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/linear_env'), args.date),
                               args.exp_title)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
        if "checkpoint_{}".format(args.checkpoint_num) in dirpath and args.exp_title == dirpath.split('/')[-3]:
            # grab the experiment name
            folder = os.path.dirname(dirpath)
            tune_name = folder.split("/")[-1]
            outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config, checkpoint_path = get_config_from_path(folder, str(args.checkpoint_num))

            ray.shutdown()
            ray.init()

            run_transfer_tests(config, checkpoint_path, 10, args.exp_title, output_path)
            visualize_adversaries(config, checkpoint_path, 100, output_path)

            for i in range(4):
                try:
                    p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                     "s3://sim2real/transfer_results/linear_env/{}/{}/{}".format(
                                                                         args.date,
                                                                         args.exp_title,
                                                                         tune_name)).split(
                        ' '))
                    p1.wait(50)
                except Exception as e:
                    print('This is the error ', e)
