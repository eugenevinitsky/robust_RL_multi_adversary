import argparse
import configparser
from copy import deepcopy
from datetime import datetime
from gym import spaces
import os
import pytz

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.mujoco.run_rollout import run_rollout, instantiate_rollout
from visualize.plot_heatmap import save_heatmap, hopper_friction_sweep, hopper_mass_sweep, cheetah_friction_sweep, cheetah_mass_sweep, ant_mass_sweep, ant_friction_sweep
import errno


def make_set_friction(friction_coef):
    def set_friction(env):
        env.model.geom_friction[:] = (env.model.geom_friction * friction_coef)[:]
        if hasattr(env, 'fric_coef'):
            env.fric_coef = friction_coef
    return set_friction

def make_set_mass(mass_coef, mass_body='pole'):
    def set_mass(env):
        bnames = env.model.body_names
        bindex = bnames.index(mass_body)
        env.model.body_mass[bindex] = (env.model.body_mass[bindex] * mass_coef)
        if hasattr(env, 'mass_coef'):
            env.fric_coef = mass_coef
    return set_mass

def make_set_mass_and_fric(friction_coef, mass_coef, mass_body='pole'):
    def set_mass(env):
        mass_bname = mass_body
        bnames = env.model.body_names
        bindex = bnames.index(mass_bname)
        env.model.body_mass[bindex] = (env.model.body_mass[bindex] * mass_coef)
        env.model.geom_friction[:] = (env.model.geom_friction * friction_coef)[:]
    return set_mass


def make_set_fric_hard(max_fric_coeff, min_fric_coeff, high_fric_idx, mass_body='pole'):
    def set_mass(env):
        env.model.geom_friction[high_fric_idx] = (env.model.geom_friction * max_fric_coeff)[high_fric_idx]
        low_fric_idx = np.ones(len(env.model.geom_friction), np.bool)
        low_fric_idx[high_fric_idx] = 0
        env.model.geom_friction[low_fric_idx] = (env.model.geom_friction * min_fric_coeff)[low_fric_idx]
    return set_mass

#hopper geoms: floor, torso, thigh, leg, foot
hopper_run_list = [
    ['base', []]
]
hopper_test_list=[
    ['friction_hard_torsolegmax_floorthighfootmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [1, 3])],
    ['friction_hard_floorthighmax_torsolegfootmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [0, 2])],
    ['friction_hard_footlegmax_floortorsothighmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [3, 4])],
    ['friction_hard_torsothighfloormax_footlegmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [0, 1, 2])],
    ['friction_hard_torsofootmax_floorthighlegmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [1, 4])],
    ['friction_hard_floorthighlegmax_torsofootmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [0, 3, 2])],
    ['friction_hard_floorfootmax_torsothighlegmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [4, 0])],
    ['friction_hard_thighlegmax_floortorsofootmin', make_set_fric_hard(max(hopper_friction_sweep), min(hopper_friction_sweep), [2, 3])],
]
num_hopper_custom_tests = len(hopper_run_list)

#cheetah geoms: ('floor', 'torso', 'head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot')
cheetah_run_list = [
    ['base', []]
]
cheetah_test_list=[
    ['friction_hard_torsoheadfthighmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [1, 2, 6])],
    ['friction_hard_floorheadfshinmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [0, 2, 7])],
    ['friction_hard_bthighbshinbfootmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [3, 4, 5])],
    ['friction_hard_floortorsoheadmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [0, 1, 2])],
    ['friction_hard_floorbshinffootmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [1, 4, 8])],
    ['friction_hard_bthighbfootffootmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [3, 5, 8])],
    ['friction_hard_bthighfthighfshinmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [3, 6, 7])],
    ['friction_hard_headfshinffootmax', make_set_fric_hard(max(cheetah_friction_sweep), min(cheetah_friction_sweep), [2, 7, 8])],
]
num_cheetah_custom_tests = len(cheetah_test_list)

#ant geoms: ('world', 'torso', 'front_left_leg', 'aux_1', 'front_right_leg', 'aux_2', 'back_leg', 'aux_3', 'right_back_leg', 'aux_4')
ant_run_list = [
    ['base', []]
]
ant_test_list=[
    ['friction_hard_flla1a3max', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [2, 3, 7])],
    ['friction_hard_torsoa1rblmax', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [1, 3, 8])],
    ['friction_hard_frla2blmax', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [4, 5, 6])],
    ['friction_hard_torsoflla1max', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [1, 2, 3])],
    ['friction_hard_flla2a4max', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [2, 5, 9])],
    ['friction_hard_frlbla4max', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [4, 6, 9])],
    ['friction_hard_frla3rblmax', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [4, 7, 8])],
    ['friction_hard_a1rbla4max', make_set_fric_hard(max(ant_friction_sweep), min(ant_friction_sweep), [3, 8, 9])],
]
num_ant_custom_tests = len(ant_test_list)

hopper_grid = np.meshgrid(hopper_mass_sweep, hopper_friction_sweep)
for mass, fric in np.vstack((hopper_grid[0].ravel(), hopper_grid[1].ravel())).T:
    hopper_run_list.append(['m_{}_f_{}'.format(mass, fric), make_set_mass_and_fric(fric, mass, mass_body="torso")])

cheetah_grid = np.meshgrid(cheetah_mass_sweep, cheetah_friction_sweep)
for mass, fric in np.vstack((cheetah_grid[0].ravel(), cheetah_grid[1].ravel())).T:
    cheetah_run_list.append(['m_{}_f_{}'.format(mass, fric), make_set_mass_and_fric(fric, mass, mass_body="torso")])

ant_grid = np.meshgrid(ant_mass_sweep, ant_friction_sweep)
for mass, fric in np.vstack((ant_grid[0].ravel(), ant_grid[1].ravel())).T:
    ant_run_list.append(['m_{}_f_{}'.format(mass, fric), make_set_mass_and_fric(fric, mass, mass_body="torso")])

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
    # high = np.array([1.0, 90.0, env.max_cart_vel, env.max_pole_vel])
    # env.observation_space = spaces.Box(low=-1 * high, high=high, dtype=env.observation_space.dtype)
    if callable(env_modifier):
        env_modifier(env)
    elif type(env) is MultiarmBandit:
        env.transfer = env_modifier
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


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir, run_list, is_test=False, render=False):

    output_file_path = os.path.join(outdir, output_file_name)
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    temp_output = [run_test.remote(test_name=list[0],
                 outdir=outdir, output_file_name=output_file_name,
                 num_rollouts=num_rollouts,
                 rllib_config=rllib_config, checkpoint=checkpoint, env_modifier=list[1], render=render) for list in run_list]
    temp_output = ray.get(temp_output)

    output_name = "mean_sweep"
    if is_test:
        output_name = "holdout_test_sweep"
    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, output_name),
              'wb') as file:
        np.savetxt(file, np.array(temp_output))

    if 'MAHopperEnv' == rllib_config['env'] and len(temp_output) > num_hopper_custom_tests:
        try:
            reward_means = np.array(temp_output)[num_hopper_custom_tests:, 0].reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))
            output_name = output_file_name + 'rew'
            save_heatmap(reward_means, hopper_mass_sweep, hopper_friction_sweep, outdir, output_name, False, 'hopper')

            step_means = np.array(temp_output)[num_hopper_custom_tests:, 2].reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))
            output_name = output_file_name + 'steps'
            save_heatmap(step_means, hopper_mass_sweep, hopper_friction_sweep, outdir, output_name, False, 'hopper')
        except:
            pass

    elif 'MACheetahEnv' == rllib_config['env']:
        if not is_test:
            reward_means = np.array(temp_output)[1:, 0].reshape(len(cheetah_mass_sweep), len(cheetah_friction_sweep))
            output_name = output_file_name + 'rew'
            save_heatmap(reward_means, cheetah_mass_sweep, cheetah_friction_sweep, outdir, output_name, False, 'cheetah')

            step_means = np.array(temp_output)[1:, 2].reshape(len(cheetah_mass_sweep), len(cheetah_friction_sweep))
            output_name = output_file_name + 'steps'
            save_heatmap(step_means, cheetah_mass_sweep, cheetah_friction_sweep, outdir, output_name, False, 'cheetah')

    elif 'MAAntEnv' == rllib_config['env']:
        if not is_test:
            reward_means = np.array(temp_output)[1:, 0].reshape(len(ant_mass_sweep), len(ant_friction_sweep))
            output_name = output_file_name + 'rew'
            save_heatmap(reward_means, ant_mass_sweep, ant_friction_sweep, outdir, output_name, False, 'ant')

            step_means = np.array(temp_output)[1:, 2].reshape(len(ant_mass_sweep), len(ant_friction_sweep))
            output_name = output_file_name + 'steps'
            save_heatmap(step_means, ant_mass_sweep, ant_friction_sweep, outdir, output_name, False, 'ant')

    elif 'MAPendulumEnv' in rllib_config['env']:
        means = np.array(temp_output)[1:, 0]
        with open('{}/{}_{}.png'.format(outdir, output_file_name, "transfer_robustness"), 'wb') as transfer_robustness:
            fig = plt.figure()
            plt.bar(np.arange(len(mass_list)), means)
            plt.title("Pendulum performance across mass values")
            plt.xticks(ticks=np.arange(len(mass_list)), labels=["{:0.2f}".format(x) for x in mass_list])
            plt.xlabel("Mass coef")
            plt.savefig(transfer_robustness)
            plt.close(fig)
    
    elif 'Bandit' in rllib_config['env']:
            means = np.array(temp_output)[:,0]
            std_devs = np.array(temp_output)[:,1]
            if len(means) > 0:
                with open('{}/{}_{}.png'.format(outdir, output_file_name, "transfer_performance"), 'wb') as transfer_robustness:
                    fig = plt.figure()
                    plt.bar(np.arange(len(means)), means)
                    plt.title("Bandit performance tests")
                    plt.xticks(ticks=np.arange(len(means)), labels=[transfer[0] for transfer in run_list])
                    plt.xlabel("Bandit test name")
                    plt.ylabel("Bandit regret")
                    plt.savefig(transfer_robustness)
                    plt.close(fig)

    num_advs = rllib_config['env_config']['advs_per_strength'] * rllib_config['env_config']['num_adv_strengths']
    if num_advs > 11:
        return
    adv_names = ["adversary{}".format(adv_num) for adv_num in range(num_advs)]
    if num_advs:
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
            adv_names = ["adversary{}" for adv_num in range(num_advs)]
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

    if rllib_config['env'] == "MAPendulumEnv":
        run_list = pendulum_run_list
    elif rllib_config['env'] == "MAHopperEnv":
        if args.run_holdout:
            run_list = hopper_test_list
        else:
            run_list = hopper_run_list
    elif rllib_config['env'] == "MACheetahEnv":
        if args.run_holdout:
            run_list = cheetah_test_list
        else:
            run_list = cheetah_run_list
    elif rllib_config['env'] == "MAAntEnv":
        if args.run_holdout:
            run_list = ant_test_list
        else:
            run_list = ant_run_list
    elif rllib_config['env'] == "MultiarmBandit":
        run_list = make_bandit_transfer_list(rllib_config['env_config']['num_arms'])

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name,
                       os.path.join(args.output_dir, date), run_list=run_list, render=args.show_images)