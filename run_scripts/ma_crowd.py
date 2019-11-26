import errno
from datetime import datetime
import os
import subprocess
import sys

import pytz
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

from algorithms.custom_ppo import KLPPOTrainer, CustomPPOPolicy
from visualize.transfer_test import run_transfer_tests
from utils.env_creator import ma_env_creator
from utils.parsers import init_parser, env_parser, ray_parser, ma_env_parser
from utils.rllib_utils import get_config_from_path

from ray.rllib.models.catalog import MODEL_DEFAULTS
from models.conv_lstm import ConvLSTM


def setup_ma_config(config):
    env = ma_env_creator(config['env_config'])
    policies_to_train = ['robot']

    policy_graphs = {'robot': (PPOTFPolicy, env.observation_space, env.action_space, {})}
    num_adversaries = config['num_adversaries']
    adv_policies = ['adversary' + str(i) for i in range(num_adversaries)]
    adversary_config = {"model": {'fcnet_hiddens': [32, 32], 'use_lstm': False}}
    policy_graphs.update({adv_policies[i]: (CustomPPOPolicy, env.adv_observation_space,
                                                 env.adv_action_space, adversary_config) for i in range(num_adversaries)})

    policies_to_train += adv_policies

    # def policy_mapping_fn(agent_id):
    #     if agent_id == 'robot':
    #         return agent_id
    #     if agent_id.startswith('adversary'):
    #         import ipdb; ipdb.set_trace()
    #         policy_choice = random.choice(adv_policies)
    #         print('the policy choice is ', policy_choice)
    #         return policy_choice
    def policy_mapping_fn(agent_id):
        return agent_id

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': policies_to_train
        }
    })


def setup_exps(args):
    parser = init_parser()
    parser = env_parser(parser)
    parser = ray_parser(parser)
    parser = ma_env_parser(parser)
    args = parser.parse_args(args)

    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = args.num_cpus
    config['gamma'] = 0.99
    config["sgd_minibatch_size"] = 500
    config["num_sgd_iter"] = 10
    config['num_adversaries'] = args.num_adv
    config['kl_diff_weight'] = args.kl_diff_weight

    config['env_config']['run'] = alg_run
    config['env_config']['policy'] = args.policy
    config['env_config']['show_images'] = args.show_images
    config['env_config']['train_on_images'] = args.train_on_images
    config['env_config']['perturb_state'] = args.perturb_state
    config['env_config']['perturb_actions'] = args.perturb_actions
    config['env_config']['num_adversaries'] = args.num_adv

    if not args.perturb_state and not args.perturb_actions:
        sys.exit('You need to select at least one of perturb actions or perturb state')

    with open(args.env_params, 'r') as file:
        env_params = file.read()

    with open(args.policy_params, 'r') as file:
        policy_params = file.read()

    config['env_config']['env_params'] = env_params
    config['env_config']['policy_params'] = policy_params

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
        config['model']['lstm_cell_size'] = 128
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
        config['vf_loss_coeff'] = 1e-4
    config['train_batch_size'] = args.train_batch_size

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = 's3://sim2real/' \
                + date + '/' + args.exp_title

    config['env'] = 'MultiAgentCrowdSimEnv'
    register_env('MultiAgentCrowdSimEnv', ma_env_creator)

    setup_ma_config(config)

    exp_dict = {
        'name': args.exp_title,
        'run_or_experiment': KLPPOTrainer,
        'checkpoint_freq': args.checkpoint_freq,
        'stop': {
            'training_iteration': args.num_iters
        },
        'config': config,
        'num_samples': args.num_samples,
    }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    return exp_dict, args


if __name__=="__main__":

    exp_dict, args = setup_exps(sys.argv[1:])
    
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init(local_mode=True)

    run_tune(**exp_dict, queue_trials=False)

    # Now we add code to loop through the results and create scores of the results
    if args.run_transfer_tests:
        date = datetime.now(tz=pytz.utc)
        date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
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
                run_transfer_tests(config, checkpoint_path, 50, args.exp_title, output_path, save_trajectory=False)
        p1 = subprocess.Popen("aws s3 sync {} {}".format(output_path, "s3://sim2real/transfer_results/{}/{}".format(date, args.exp_title)).split(' '))
        p1.wait()