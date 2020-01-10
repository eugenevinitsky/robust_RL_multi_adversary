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
    # how many adversaries we will test with
    num_test_adversaries = options_dict['num_test_adversaries']
    # how many adversaries we actually train with
    train_adversary_num = options_dict['train_adversary_num']
    # we track our score for each adversary, the prediction error, and whether we guess that adversary correct
    # The plus 1 is because the last index is with the adversaries off. This lets us see what we are guessing
    # when there actually isn't any adversary
    results_dict['adversary_rew'] = np.zeros(num_test_adversaries + 1)
    results_dict['prediction_error'] = np.zeros(num_test_adversaries + 1)
    results_dict['adversary_guess'] = np.zeros(num_test_adversaries + 1)

    # here we track the magnitude of torques across time as well as the magnitude of the state error across time
    results_dict['action_time'] = np.zeros((num_test_adversaries + 1, env.horizon + 1))
    if env.guess_next_state:
        results_dict['state_pred_time'] = np.zeros((num_test_adversaries + 1, env.horizon + 1))

    # we also plot which adversary we guess for any given adversary
    if env.guess_adv:
        results_dict['guess_adv_time'] = np.zeros((num_test_adversaries + 1, env.horizon + 1))
        # the column index is one less because we would never guess an adversary larger than the number we trained with
        results_dict['guess_grid'] = np.zeros((num_test_adversaries + 1, train_adversary_num))

    # Visualize the evolution of state per adversary. We only plot the last trajectory
    shape_scaling = 1
    if hasattr(env, 'num_concat_states'):
        shape_scaling = env.num_concat_states
    # we subtract off 1 * shape_scaling to remove the previous actions from the observation space
    results_dict['state_time'] = np.zeros((num_test_adversaries + 1, env.horizon + 1,
                                           int((env.observation_space.shape[0] - 1 * shape_scaling) / shape_scaling)))
    return results_dict


def done_func(env, results_dict):
    results_dict['prediction_error'][env.curr_adversary] = np.linalg.norm(env.state_error / env.horizon)
    results_dict['adversary_guess'][env.curr_adversary] += env.num_correct_guesses / env.horizon


def step_func(multi_obs, action_dict, logits_dict, results_dict, env):
    results_dict['guess_grid'][env.curr_adversary, int(action_dict['agent0'][-1])] += 1
    results_dict['action_time'][env.curr_adversary, env.step_num] += np.abs(action_dict['agent0'][0])
    if env.guess_next_state:
        results_dict['state_pred_time'][env.curr_adversary, env.step_num] += np.linalg.norm(env.curr_state_error)

    if env.guess_adv:
        results_dict['guess_adv_time'][env.curr_adversary, env.step_num] += int(action_dict['agent0'][-1])

    results_dict['state_time'][env.curr_adversary, env.step_num] = env._get_obs()


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
    plt.figure()
    xvals = ['adv{}'.format(i) for i in range(results_dict['adversary_rew'].shape[0] - 1)]
    xvals.append('No adv')
    sns.barplot(xvals, results_dict['adversary_rew'])
    output_str = '{}/{}'.format(outdir, 'adversary_rewards.png')
    plt.title('Adversary vs. Reward')
    plt.xlabel('Adversary number')
    plt.ylabel('Reward')
    plt.savefig(output_str)
    plt.close()

    # Plot the histogram of prediction error
    plt.figure()
    sns.barplot(xvals, results_dict['prediction_error'] / num_rollouts)
    output_str = '{}/{}'.format(outdir, 'prediction_error.png')
    plt.title('Adversary vs. Prediction Error')
    plt.xlabel('Adversary number')
    plt.ylabel('Prediction error')
    plt.savefig(output_str)
    plt.close()

    # Plot the histogram of number of correct guesses
    plt.figure()
    sns.barplot(xvals[:-1], results_dict['adversary_guess'][:-1] / num_rollouts)
    output_str = '{}/{}'.format(outdir, 'adversary_guess.png')
    plt.title('Adversary number vs. percent correct guesses')
    plt.xlabel('Adversary number')
    plt.ylabel('Percentage of correct guesses')
    plt.savefig(output_str)
    plt.close()

    if 'guess_grid' in results_dict.keys():
        guess_grid = results_dict['guess_grid']
        # Plot which adversary we guessed for any given adversaries
        plt.figure()
        # increasing the row index implies moving down on the y axis
        sns.heatmap(guess_grid / num_rollouts, yticklabels=list(range(guess_grid.shape[0])),
                    xticklabels=list(range(guess_grid.shape[1])))
        plt.ylabel('Current adversary')
        plt.xlabel('Adversary guess')
        plt.title('The extra row is for adversaries turned off')
        output_str = '{}/{}'.format(outdir, 'guess_grid.png')
        plt.savefig(output_str)
        plt.close()

    if 'state_pred_time' in results_dict.keys():
        for i in range(results_dict['state_pred_time'].shape[0]):
            if i < results_dict['state_pred_time'].shape[0] - 1:
                adv_str = str(i)
            else:
                adv_str = 'no_adversary'
            plt.figure()
            plt.plot(results_dict['state_pred_time'][i, :] / num_rollouts)
            plt.xlabel('step number')
            plt.ylabel('average state error')
            plt.title('Time vs. norm of adversary error, adversary {}'.format(adv_str))
            output_str = '{}/{}'.format(outdir, 'state_err_v_time_adv_{}.png'.format(adv_str))
            plt.savefig(output_str)
            plt.close()

    if 'guess_adv_time' in results_dict.keys():
        for i in range(results_dict['guess_adv_time'].shape[0]):
            if i < results_dict['state_pred_time'].shape[0] - 1:
                adv_str = str(i)
            else:
                adv_str = 'no_adversary'
            plt.figure()
            plt.plot(results_dict['guess_adv_time'][i, :] / num_rollouts)
            plt.xlabel('step number')
            plt.ylabel('average adversary guess')
            plt.title('Time vs. norm of adversary {} error'.format(adv_str))
            output_str = '{}/{}'.format(outdir, 'guess_v_time_adv_{}.png'.format(adv_str))
            plt.savefig(output_str)
            plt.close()

    for i in range(results_dict['action_time'].shape[0]):
        if i < results_dict['state_pred_time'].shape[0] - 1:
            adv_str = str(i)
        else:
            adv_str = 'no_adversary'
        plt.figure()
        plt.plot(results_dict['action_time'][i, :] / num_rollouts)
        plt.xlabel('step number')
        plt.ylabel('action magnitude')
        plt.title('Time vs. norm of action, adversary {}'.format(adv_str))
        output_str = '{}/{}'.format(outdir, 'action_v_time_{}.png'.format(adv_str))
        plt.savefig(output_str)
        plt.close()

    ylabels = ['x', 'y', 'theta_dot']
    for i in range(results_dict['state_time'].shape[0]):
        if i < results_dict['state_time'].shape[0] - 1:
            adv_str = str(i)
        else:
            adv_str = 'no_adversary'
        for j in range(results_dict['state_time'].shape[-1]):
            plt.figure()
            plt.plot(results_dict['state_time'][i, :, j])
            plt.xlabel('step number')
            plt.ylabel(ylabels[j])
            plt.title('Time vs. state {} evolution, adversary {}'.format(ylabels[j], adv_str))
            output_str = '{}/{}'.format(outdir, 'state_{}_v_time_{}.png'.format(ylabels[j], adv_str))
            plt.savefig(output_str)
        plt.figure()
        plt.plot(results_dict['state_time'][i, :, 0], results_dict['state_time'][i, :, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Phase evolution, adversary {}'.format(adv_str))
        output_str = '{}/{}'.format(outdir, 'phase_evolution_{}.png'.format(adv_str))
        plt.savefig(output_str)


def visualize_model_perf(rllib_config, checkpoint, num_test_adversaries, num_rollouts, outdir):
    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = \
        instantiate_rollout(rllib_config, checkpoint)

    options_dict = {'num_test_adversaries': num_test_adversaries, 'train_adversary_num': env.num_adversaries}

    env.num_adversaries = num_test_adversaries

    results_dict = dict_func(env, options_dict)
    for curr_adversary in range(num_test_adversaries):
        env.curr_adversary = curr_adversary
        new_results = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                                   state_init, action_init, num_rollouts,
                                   None, step_func, done_func, options_dict, results_dict)
        results_dict['adversary_rew'][env.curr_adversary] = np.mean(new_results['rewards'])

    # now do one run with the adversary off
    env.curr_adversary += 1
    env.has_adversary = False
    new_results = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                              state_init, action_init, num_rollouts,
                              None, step_func, done_func, options_dict, results_dict)
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