"""Runs the training script and computes transfer scores."""

from datetime import datetime
import json
import pytz
import os
import subprocess
import sys

import algorithms.baselines_her.multiagent_her.experiment.config as config
from baselines.common import tf_util
from run_scripts.mujoco.run_adv_mujoco import get_parser, get_env_config
from algorithms.baselines_her.multiagent_her.cmd_util import common_arg_parser, make_env
from algorithms.baselines_her.run import parse_cmdline_kwargs
from algorithms.baselines_her.multiagent_her.transfer_tests import run_transfer_tests

date = datetime.now(tz=pytz.utc)
date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
save_path = "/Users/eugenevinitsky/her_results/models/05-23-2020/push_env_0_adv"
log_path = "/Users/eugenevinitsky/her_results/log_data"

def main(args):
    # # NOTE: 5000 is one epoch
    # cmd = "mpirun -np 4 python -m run --alg=multiagent_her --env=MAFetchPushEnv " \
    #                       "--num_timesteps=10000 --num_adv 5 --adv_all_actions " \
    #                       "--adv_strength 0.25 --num_adv_strengths 1 --advs_per_strength 5 " \
    #                       "--return_all_obs --log_path {} " \
    #                       "--save_path {}".format(log_path, save_path)
    # cmd_list = cmd.split(' ')
    # print(cmd)
    # print(cmd_list)
    # p1 = subprocess.Popen(cmd_list)
    # p1.wait()

    # now we load up the model and run the transfer tests
    # build the model
    params = config.DEFAULT_PARAMS
    # TODO(@ev) put in better ones
    temp_config = {'env_config': {}}
    arg_parser = common_arg_parser()
    arg_parser = get_parser(arg_parser)
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    passed_config = get_env_config(args, temp_config)
    env_id = "MAFetchPushEnv"
    env = make_env(env_id, "", config=passed_config,
                   flatten_dict_observations=False)
    env.env.env.should_render = True
    env.adversary_range = 0
    env_name = "MAFetchPushEnv-v1"
    params['env_name'] = env_name
    params = config.prepare_params(env, params)
    # params['rollout_batch_size'] = env.num_envs
    # TODO(@ev) put back
    params['rollout_batch_size'] = 1

    dims = config.configure_dims(params)
    model = config.configure_ddpg(dims=dims, params=params, clip_return=False, name='agent')

    tf_util.load_variables(save_path)
    results_dir = os.path.join(os.path.dirname(save_path), 'results/' + os.path.basename(save_path))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    run_transfer_tests(results_dir, env, env_id, model, 10)

if __name__ == '__main__':
    main(sys.argv[1:])
