"""Run transfer tests on the cluster and upload them to AWS"""

import argparse
import errno
import os
import subprocess
from datetime import date
import ray

from visualize.mujoco.transfer_tests import run_transfer_tests
from visualize.mujoco.action_sampler import sample_actions
from utils.rllib_utils import get_config_from_path

parser = argparse.ArgumentParser()
parser.add_argument('exp_title', type=str)
parser.add_argument('checkpoint_num', type=int)
parser.add_argument('date', type=str, help='A date in M-DD-YYYY format')


args = parser.parse_args()


exp_date = args.date

ray.init()

output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results'), date.today().strftime("%m-%d-%Y")), args.exp_title)
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

# now run sync_s3
os.chmod(os.path.expanduser("~/adversarial_sim2real/run_scripts/s3_sync.sh"), 0o775)
p1 = subprocess.Popen(os.path.expanduser("~/adversarial_sim2real/run_scripts/s3_sync.sh {}".format(exp_date)).split(' '))
p1.wait()

for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/s3_test")):
    if "checkpoint_{}".format(args.checkpoint_num) in dirpath and args.exp_title in dirpath:
        # grab the experiment name
        folder = os.path.dirname(dirpath)
        tune_name = folder.split("/")[-1]
        outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config, checkpoint_path = get_config_from_path(folder, str(args.checkpoint_num))
        test_list = []
        if config['env'] == "MALerrelPendulumEnv":
            from visualize.pendulum.transfer_tests import pendulum_run_list

            lerrel_run_list = pendulum_run_list
        elif config['env'] == "MALerrelHopperEnv":
            from visualize.pendulum.transfer_tests import hopper_run_list

            lerrel_run_list = hopper_run_list
        elif config['env'] == "MALerrelCheetahEnv":
            from visualize.pendulum.transfer_tests import cheetah_run_list
            lerrel_run_list = cheetah_run_list

        elif config['env'] == "MAHopperEnv":
            from visualize.mujoco.transfer_tests import hopper_run_list, hopper_test_list
            lerrel_run_list = hopper_run_list
            test_list  = hopper_test_list
        elif config['env'] == "MACheetahEnv":
            from visualize.mujoco.transfer_tests import cheetah_run_list, cheetah_test_list
            lerrel_run_list = cheetah_run_list
            test_list = cheetah_test_list
        elif config['env'] == "MAAntEnv":
            from visualize.mujoco.transfer_tests import ant_run_list, ant_test_list
            lerrel_run_list = ant_run_list
            test_list = ant_test_list

        elif config['env'] == "MAFetchReachEnv":
            from visualize.mujoco.transfer_tests import fetch_reach_run_list
            lerrel_run_list = fetch_reach_run_list
        elif config['env'] == "MAFetchPushEnv":
            from visualize.mujoco.transfer_tests import fetch_push_run_list
            lerrel_run_list = fetch_push_run_list

        ray.shutdown()
        ray.init()
        run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=lerrel_run_list)
        if len(test_list) > 0:
            run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=test_list, is_test=True)

        sample_actions(config, checkpoint_path, 10000, output_path)

        # visualize_adversaries(config, checkpoint_path, 10, 100, output_path)
        for i in range(4):
            try:
                p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                                 "s3://sim2real/transfer_results/adv_robust/{}/{}/{}".format(
                                                                     date.today().strftime("%m-%d-%Y"),
                                                                     args.exp_title,
                                                                     tune_name)).split(
                    ' '))
                p1.wait(50)
            except Exception as e:
                print('This is the error ', e)