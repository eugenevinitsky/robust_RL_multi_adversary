import argparse
import configparser
from copy import deepcopy
from datetime import datetime
from gym import spaces
import os
import pytz
import seaborn as sns

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
COLORS = ["#95d0ff", "#966bff", "#ff6ad5", "#ffa58b", "#ff6a8b"]

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.pendulum.run_rollout import run_rollout, instantiate_rollout
import errno

# TODO(@evinitsky) put back
engine_strength_grid = np.linspace(7.5, 20.0, 20)
engine_strength_hard = np.linspace(7.5, 11, 20)


def make_set_engine_strength(engine_strength):
    def set_engine_strength(env):
        env.main_engine_power = engine_strength
    return set_engine_strength

engine_strength_run_list = [['str_{}'.format(strength), make_set_engine_strength(strength)]
                            for strength in engine_strength_grid]
engine_strength_run_list_hard =[['str_{}_hard'.format(strength), make_set_engine_strength(strength)]
                                for strength in engine_strength_hard]


def get_plot_config(environment):
    if environment == 'lunar' or 'Lunar' in environment:
        return {
            'metrics': ['ref_learning_curve_{}', 'hard_learning_curve_{}', 'rand_learning_curve_{}'],
            'solved': 200,
            'xlim': (7.5, 20.0),
            'ylim': (0, 330),
            'start_index': 0,
            'environment': environment,
            # 'labels': ['baseline', 'UDR', 'oracle', 'ADR (ours)'],
            'labels': ['Oracle', 'Baseline', 'UDR', 'ADR (ours)'],
            'title': 'Generalization Results (LunarLander)',
            # 'title': 'Oracle vs. UDR (LunarLander)',
            'dimensions': 1,
            'colors': COLORS,
            'legend_loc': 'lower right',
            'x_label': 'Main Engine Strength (MES)',
            'y_label': 'Average Reward'
        }
    elif environment == 'lunar2':
        return {
            'metrics': ['ref_learning_curve_{}', 'hard_learning_curve_{}'],
            'solved': 200,
            'xlim': (7.5, 20.0),
            'ylim': (-100, 330),
            'start_index': 0,
            'environment': environment,
            'labels': ['$Baseline$', '$UDR$', '$ADR (ours)$'],
            'title': ['Learning Curve (LL), Reference Env.', 'Learning Curve (LL), Hard Env.'],
            'dimensions': 1,
            'colors': [COLORS[1], COLORS[2], COLORS[0]],
            'legend_loc': 'best',
            'x_label': 'Main Engine Strength (MES)',
            'y_label': 'Average Reward'
        }
    elif environment == 'lunarbootstrap':
        return {
            'metrics': ['ref_learning_curve_{}'],
            'solved': 200,
            'xlim': (7.5, 11),
            'ylim': (-150, 330),
            'start_index': 0,
            'environment': environment,
            'labels': ['$ADR(boostrapped)$', '$ADR(original)$'],
            'title': ['Bootstrapped ADR (LL)'],
            'dimensions': 1,
            'colors': [COLORS[1], COLORS[0]],
            'legend_loc': 'lower right',
            'x_label': 'Main Engine Strength (MES)',
            'y_label': 'Average Reward'
        }


def gen_plot(config, file_path, data, title=None, learning_curve=False):
    plt.figure(figsize=(6, 5))

    plt.title(config['title'] if not title else title)
    plt.xlabel(config['x_label'])
    plt.ylabel(config['y_label'])

    plt.ylim(*config['ylim'])
    if config['solved']:
        # plt.axhline(config['solved'], color=COLORS[4], linestyle='--', label='$[Solved]$') # only for figure 1
        plt.axhline(config['solved'], color=COLORS[3], linestyle='--', label='$[Solved]$')

    # colors = config['colors'][::-1][1:] # only for figure 1
    colors = config['colors']
    for i, entry in enumerate(data):
        timesteps, averaged_curve, sigma, convergence = entry
        sns.lineplot(timesteps,
                     averaged_curve,
                     c=colors[i])
                     # label=config['labels'][i])
        if convergence is not None:
            plt.plot([timesteps[-1], timesteps[-1] + 0.5],
                     [averaged_curve.values[-1], averaged_curve.values[-1]],
                     color=colors[i],
                     linestyle='--')

        plt.fill_between(x=timesteps,
                         y1=averaged_curve + sigma,
                         y2=averaged_curve - sigma,
                         facecolor=colors[i],
                         alpha=0.1)
    if learning_curve:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.legend(loc=config['legend_loc'], frameon=True, framealpha=0.5)
    plt.grid(b=False)

    # plt.show()

    plt.savefig(fname=file_path,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


def reset_env(env, num_active_adv=0):
    """Undo parameters that need to be off"""
    if hasattr(env, 'domain_randomization'):
        env.domain_randomization = False
    if num_active_adv > 0:
        env.adversary_range = env.advs_per_strength * env.num_adv_strengths


@ray.remote(memory=1500 * 1024 * 1024)
def run_test(test_name, outdir, output_file_name, num_rollouts,
             rllib_config, checkpoint, env_modifier, render, adv_num=None):
    """Run an individual transfer test

    Parameters
    ----------
    test_name: (str)
        Name of the test we are running. Used to find which env param to set to True
    is_env_config: (bool)
        If true we write the param into the env config so that it is actually updated correctly since some of the
        env configuration is done through the env_params.config
    config_value: (bool or str or None)
        This is the value we will insert into the env config. For example, True, or "every_step" or so on
    params_name: (str or None)
        If is_env_config is true, this is the key into which we will store things
    params_value: (bool or str or None)
        If is_env_config is true, this is the value we will store into the env_params.config
    outdir: (str)
        Directory results are saved to
    output_file_name: (str)
        Prefix string for naming the files. Used to uniquely identify experiments
    save_trajectory: (bool)
        If true, a video will be saved for this rollout
    show_images: (bool)
        If true, a render of the rollout will be displayed on your machine
    num_rollouts: (int)
        How many times to rollout the test. Increasing this should yield more stable results
    rllib_config: (dict)
        Passed rllib config
    checkpoint: (int)
        Number of the checkpoint we want to replay
    """
    # First compute a baseline score to compare against
    print(
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "Running the {} score!\n"
        "**********************************************************\n"
        "**********************************************************\n"
        "**********************************************************".format(test_name)
    )

    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = instantiate_rollout(rllib_config, checkpoint)
    if adv_num:
        reset_env(env, 1)
    if callable(env_modifier):
        env_modifier(env)
    elif len(env_modifier) > 0:
        setattr(env, env_modifier[0], env_modifier[1])
    rewards, step_num = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                                 state_init, action_init, num_rollouts, render, adv_num)

    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, test_name),
              'wb') as file:
        np.savetxt(file, rewards, delimiter=', ')

    print('The average reward for task {} is {}'.format(test_name, np.mean(rewards)))
    print('The average step length for task {} is {}'.format(test_name, np.mean(step_num)))

    return np.mean(rewards), np.std(rewards), np.mean(step_num), np.std(step_num)


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir, render=False):

    output_file_path = os.path.join(outdir, output_file_name)
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    run_list = engine_strength_run_list
    temp_output = [run_test.remote(test_name=list[0],
                 outdir=outdir, output_file_name=output_file_name,
                 num_rollouts=num_rollouts,
                 rllib_config=rllib_config, checkpoint=checkpoint, env_modifier=list[1], render=render) for list in run_list]
    temp_output = ray.get(temp_output)
    data = [[engine_strength_grid, np.array(temp_output)[:, 0], np.array(temp_output)[:, 1], None]]
    gen_plot(get_plot_config(rllib_config['env']), '{}/{}_{}'.format(outdir, output_file_name, "grid_search.png"),
             data, 'engine_strength')

    run_list = engine_strength_run_list_hard
    temp_output = [run_test.remote(test_name=list[0],
                 outdir=outdir, output_file_name=output_file_name,
                 num_rollouts=num_rollouts,
                 rllib_config=rllib_config, checkpoint=checkpoint, env_modifier=list[1], render=render) for list in run_list]
    temp_output = ray.get(temp_output)
    data = [[engine_strength_hard, np.array(temp_output)[:, 0], np.array(temp_output)[:, 1], None]]
    gen_plot(get_plot_config(rllib_config['env']), '{}/{}_{}'.format(outdir, output_file_name, "hard_grid_search.png"),
             data, 'engine_strength')

    # Now save the adversary results if we have any
    num_advs = rllib_config['env_config']['advs_per_strength'] * rllib_config['env_config']['num_adv_strengths']
    adv_names = ["adversary{}".format(adv_num) for adv_num in range(num_advs)]
    if num_advs > 0:
        temp_output = [run_test.remote(test_name="adversary{}".format(adv_num),
                    outdir=outdir, output_file_name=output_file_name,
                    num_rollouts=num_rollouts,
                    rllib_config=rllib_config, checkpoint=checkpoint, render=render, env_modifier=[], adv_num=adv_num)
                    for adv_num in range(num_advs)]
        temp_output = ray.get(temp_output)

        with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, "with_adv_mean_sweep"),
                'wb') as file:
            np.savetxt(file, np.array(temp_output))

        with open('{}/{}_{}'.format(outdir, output_file_name, "adv_scores.png"),
                'wb') as file:
            means = np.array(temp_output)[:,0]
            fig = plt.figure()
            plt.bar(np.arange(num_advs), means)
            plt.title("Scores under each adversary")
            plt.xticks(np.arange(num_advs), adv_names)
            plt.xlabel("Adv name")
            plt.savefig(file)
            plt.close(fig)

        with open('{}/{}_{}'.format(outdir, output_file_name, "adv_steps.png"),
                'wb') as file:
            steps = np.array(temp_output)[:,2]
            adv_names = ["adversary{}".format(adv_num) for adv_num in range(num_advs)]
            fig = plt.figure()
            plt.bar(np.arange(num_advs), steps)
            plt.title("Steps under each adversary")
            plt.xticks(np.arange(num_advs), adv_names)
            plt.xlabel("Adv name")
            plt.savefig(file)
            plt.close(fig)

if __name__ == '__main__':

    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    output_path = os.path.expanduser('~/transfer_results/')

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--output_dir', type=str, default=output_path,
                        help='')
    parser.add_argument('--run_holdout',  action='store_true', default=False, help='If true, run holdout tests')

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name,
                       os.path.join(args.output_dir, date), render=args.show_images)