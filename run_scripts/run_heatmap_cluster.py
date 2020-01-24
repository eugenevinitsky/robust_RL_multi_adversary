"""Script to use when the transfer test process fails on the cluster for SOME reason"""
import argparse
import errno
import psutil
import os
import subprocess

import ray


from utils.rllib_utils import get_config_from_path
from visualize.pendulum.transfer_tests import run_transfer_tests
from visualize.pendulum.action_sampler import sample_actions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('date', type=str, help='date in format m-d-y')
    parser.add_argument('exp_title', type=str, help='name of experiment you ran')
    parser.add_argument('num_iters', type=int, help='How many iterations the exp was run for')
    args = parser.parse_args()

    output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/adv_robust'), args.date), args.exp_title)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
        if "checkpoint_{}".format(args.num_iters) in dirpath:
            # grab the experiment name
            folder = os.path.dirname(dirpath)
            tune_name = folder.split("/")[-1]
            outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.expanduser(os.path.join(outer_folder, "visualize/transfer_test.py"))
            config, checkpoint_path = get_config_from_path(folder, str(args.num_iters))

            # TODO(@ev) gross find somewhere else to put this

            if config['env'] == "MALerrelPendulumEnv":
                from visualize.pendulum.transfer_tests import pendulum_run_list
                lerrel_run_list = pendulum_run_list
            elif config['env'] == "MALerrelHopperEnv":
                from visualize.pendulum.transfer_tests import hopper_run_list
                lerrel_run_list = hopper_run_list
            elif config['env'] == "MALerrelCheetahEnv":
                from visualize.pendulum.transfer_tests import cheetah_run_list
                lerrel_run_list = cheetah_run_list

            ray.shutdown()
            ray.init()
            run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=lerrel_run_list)
            sample_actions(config, checkpoint_path, min(2 * args.train_batch_size, 20000), output_path)

            if args.use_s3:
                # visualize_adversaries(config, checkpoint_path, 10, 100, output_path)
                for i in range(4):
                    try:
                        p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                         "s3://sim2real/transfer_results/adv_robust/{}/{}/{}".format(
                                                                             args.date,
                                                                             args.exp_title,
                                                                             tune_name)).split(
                            ' '))
                        p1.wait(50)
                    except Exception as e:
                        print('This is the error ', e)
