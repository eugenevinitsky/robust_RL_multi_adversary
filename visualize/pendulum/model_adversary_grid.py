import argparse
import errno
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns; sns.set()

from visualize.pendulum.run_rollout import instantiate_rollout, run_rollout
from utils.parsers import replay_parser
from utils.rllib_utils import get_config

"""Run through all of the model based adversaries and compute the score per adversary"""


def dict_func(env, options_dict):
    # figure out how many adversaries you have and initialize their grids
    results_dict = {}
    num_test_adversaries = options_dict['num_test_adversaries']
    # we track our score for each adversary, the prediction error, and whether we guess that adversary correct
    results_dict['adversary_rew'] = np.zeros(num_test_adversaries)
    results_dict['prediction_error'] = np.zeros(num_test_adversaries)
    results_dict['adversary_guess'] = np.zeros(num_test_adversaries)
    return results_dict


def done_func(env, results_dict):
    results_dict['prediction_error'][env.curr_adversary] = np.linalg.norm(env.state_error / env.horizon)
    results_dict['adversary_guess'][env.curr_adversary] += env.num_correct_guesses / env.horizon


def on_result(results_dict, outdir, num_rollouts):
    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # Plot the histogram of rewards
    import ipdb; ipdb.set_trace()
    plt.figure()
    sns.distplot(results_dict['adversary_rew'])
    output_str = '{}/{}'.format(outdir, 'adversary_rewards.png')
    plt.savefig(output_str)

    # Plot the histogram of rewards
    plt.figure()
    sns.distplot(results_dict['prediction_error'] / num_rollouts)
    output_str = '{}/{}'.format(outdir, 'prediction_error.png')
    plt.savefig(output_str)

    # Plot the histogram of rewards
    plt.figure()
    sns.distplot(results_dict['adversary_guess'] / num_rollouts)
    output_str = '{}/{}'.format(outdir, 'adversary_guess.png')
    plt.savefig(output_str)


def visualize_model_perf(rllib_config, checkpoint, num_test_adversaries, num_rollouts, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    env.num_adversaries = num_test_adversaries

    options_dict = {'num_test_adversaries': num_test_adversaries}

    results_dict = dict_func(env, options_dict)
    for curr_adversary in range(num_test_adversaries):
        env.curr_adversary = curr_adversary
        new_results = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                                   state_init, action_init, num_rollouts,
                                   None, None, done_func, options_dict, results_dict)
        results_dict['adversary_rew'][env.curr_adversary] = np.mean(new_results['rewards'])
    on_result(results_dict, outdir, num_rollouts)


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    parser.add_argument('--num_test_adversaries', type=int, default=20,
                        help='How many adversaries to actually test with')
    parser.add_argument('--output_dir', type=str, default='transfer_results',
                        help='Directory to output the files into')
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    rllib_config, checkpoint = get_config(args)
    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})

    ray.init(num_cpus=args.num_cpus)

    visualize_model_perf(rllib_config, checkpoint, args.num_test_adversaries, args.num_rollouts, args.output_dir)


if __name__ == '__main__':
    main()