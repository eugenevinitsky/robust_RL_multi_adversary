from datetime import datetime
import errno
import os
import subprocess
import sys

import pytz
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from visualize.transfer_test import run_transfer_tests
from utils.env_creator import env_creator, construct_config
from utils.parsers import init_parser, env_parser, ray_parser
from utils.rllib_utils import get_config_from_path

from ray.rllib.models.catalog import MODEL_DEFAULTS
from models.conv_lstm import ConvLSTM


def setup_exps(args):
    parser = init_parser()
    parser = env_parser(parser)
    parser = ray_parser(parser)
    args = parser.parse_args(args)

    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = args.num_cpus
    config["sgd_minibatch_size"] = 500
    config['train_batch_size'] = args.train_batch_size
    config["num_sgd_iter"] = 10
    config['gamma'] = 0.99
    config['lr'] = 5e-5

    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()

    config['env_config'] = construct_config(env_params, policy_params, args)
    config['env_config']['run'] = alg_run

    # pick out the right model
    if args.train_on_images:
        # register the custom model
        ModelCatalog.register_custom_model("rnn", ConvLSTM)

        conv_filters = [
            [32, [3, 3], 2],
            [32, [3, 3], 2],
        ]
        config['model'] = MODEL_DEFAULTS.copy()

        config['model']['conv_activation'] = 'relu'
        # The first list is hidden layers before the LSTM, the second list is hidden layers after the LSTM.
        config['model']['custom_options']['fcnet_hiddens'] = [[32, 32], []]
        # If this is true we concatenate the actions onto the network post-convolution
        config['model']['custom_options']['use_prev_action'] = True
        config['model']['conv_filters'] = conv_filters
        config['model']['custom_model'] = "rnn"

        config['vf_share_layers'] = True
    else:
        config['model']['fcnet_hiddens'] = [64, 64]
        config['model']['use_lstm'] = True
        config['model']['lstm_use_prev_action_reward'] = True
        config['model']['lstm_cell_size'] = 128
        config['vf_share_layers'] = True
        config['vf_loss_coeff'] = 1e-3

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/' \
                + date + '/' + args.exp_title
    config['env'] = 'CrowdSim'
    register_env('CrowdSim', env_creator)

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': alg_run,
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }

    return exp_dict, args


if __name__ == "__main__":

    exp_dict, args = setup_exps(sys.argv[1:])

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/' \
                + date + '/' + args.exp_title
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()

    run_tune(**exp_dict, queue_trials=False)

    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        output_path = os.path.join(os.path.join(os.path.expanduser('~/transfer_results'), date), args.exp_title)
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
            # TODO(@evinitsky) this is pretty brittle, we should just remove the test folder from sim2real
            if "checkpoint_{}".format(args.num_iters) in dirpath and 'test' not in dirpath:
                # grab the experiment name
                folder = os.path.dirname(dirpath)
                tune_name = folder.split("/")[-1]
                outer_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.expanduser(os.path.join(outer_folder, "visualize/transfer_test.py"))
                config, checkpoint_path = get_config_from_path(folder, str(args.num_iters))
                run_transfer_tests(config, checkpoint_path, 500, args.exp_title, output_path, save_trajectory=False)
        p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path, "s3://sim2real/transfer_results/{}/{}".format(date, args.exp_title)).split(' '))
        p1.wait()