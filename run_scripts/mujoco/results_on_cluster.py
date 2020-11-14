import errno
from datetime import datetime
import os
import subprocess
import sys

import pytz
import ray

from visualize.mujoco.transfer_tests import run_transfer_tests
from utils.rllib_utils import get_config_from_path
from utils.parsers import init_parser, ray_parser, ma_env_parser


if __name__ == '__main__':
    parser = init_parser()
    parser = ray_parser(parser)
    parser = ma_env_parser(parser)
    parser.add_argument('results_dir', type=str,
                        help='Where to look for results')
    parser.add_argument('upload_dir', type=str,
                        help='Where to upload results')

    args = parser.parse_args()

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")

    if args.multi_node:
      ray.init(redis_address='localhost:6379')
    elif args.local_mode:
      ray.init(local_mode=True)
    else:
      ray.init()

    # Now we add code to loop through the results and create scores of the results
    for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser(args.results_dir)):
      # if "checkpoint_{}".format(args.num_iters) in dirpath:
      if "checkpoint" in dirpath:

        import ipdb; ipdb.set_trace()
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results/adv_robust'), date),
                                   args.upload_dir)
        if not os.path.exists(output_path):
          try:
            os.makedirs(output_path)
          except OSError as exc:
            if exc.errno != errno.EEXIST:
              raise

        # grab the experiment name
        folder = os.path.dirname(dirpath)
        tune_name = folder.split("/")[-1]
        outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.expanduser(os.path.join(outer_folder, "visualize/transfer_test.py"))
        config, checkpoint_path = get_config_from_path(folder, dirpath.split('_')[-1])

        test_list = []
        # TODO(@ev) gross find somewhere else to put this
        if config['env'] == "MAPendulumEnv":
          from visualize.mujoco.transfer_tests import pendulum_run_list

          run_list = pendulum_run_list
        elif config['env'] == "MAHopperEnv":
          from visualize.mujoco.transfer_tests import hopper_run_list, hopper_test_list

          run_list = hopper_run_list
          test_list = hopper_test_list
        elif config['env'] == "MACheetahEnv":
          from visualize.mujoco.transfer_tests import cheetah_run_list, cheetah_test_list

          run_list = cheetah_run_list
          test_list = cheetah_test_list
        elif config['env'] == "MAAntEnv":
          from visualize.mujoco.transfer_tests import ant_run_list, ant_test_list

          run_list = ant_run_list
          test_list = ant_test_list
        elif config['env'] == "MABallInCupEnv":
          from visualize.mujoco.transfer_tests import cup_run_list

          run_list = cup_run_list
        elif config['env'] == "MAFingerEnv":
          from visualize.mujoco.transfer_tests import finger_run_list

          run_list = finger_run_list

        ray.shutdown()
        ray.init()
        run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=run_list)
        if len(test_list) > 0:
          run_transfer_tests(config, checkpoint_path, 20, args.exp_title, output_path, run_list=test_list, is_test=True)


        if args.use_s3:
          # visualize_adversaries(config, checkpoint_path, 10, 100, output_path)
          for i in range(4):
            try:
              p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path,
                                                               's3://sim2real/transfer_results/adv_robust/' \
                                                               + date + '/' + args.upload_dir
                                                               ).split(
                ' '))
              p1.wait(50)
            except Exception as e:
              print('This is the error ', e)