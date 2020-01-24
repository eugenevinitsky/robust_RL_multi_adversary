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
from visualize.pendulum.run_rollout import run_rollout, instantiate_rollout
from visualize.plot_heatmap import save_heatmap, hopper_friction_sweep, hopper_mass_sweep, cheetah_friction_sweep, cheetah_mass_sweep
import errno


def make_set_friction(friction_coef):
    def set_friction(env):
        env.model.geom_friction[:] = (env.model.geom_friction * friction_coef)[:]
    return set_friction

def make_set_mass(mass_coef, mass_body='pole'):
    def set_mass(env):
        bnames = env.model.body_names
        bindex = bnames.index(mass_body)
        env.model.body_mass[bindex] = (env.model.body_mass[bindex] * mass_coef)
    return set_mass

def make_set_mass_and_fric(friction_coef, mass_coef, mass_body='pole'):
    def set_mass(env):
        mass_bname = 'torso'
        bnames = env.model.body_names
        bindex = bnames.index(mass_bname)
        env.model.body_mass[bindex] = (env.model.body_mass[bindex] * mass_coef)
        env.model.geom_friction[:] = (env.model.geom_friction * friction_coef)[:]
    return set_mass
    

# test name, is_env_config, config_value, params_name, params_value
run_list = [
    ['base', []],
    ['friction', ['friction', True]],
    ['gaussian_action_noise', ['add_gaussian_action_noise', True]],
    ['gaussian_state_noise', ['add_gaussian_state_noise', True]]
]

# test name, is_env_config, config_value, params_name, params_value
mass_list = [0.5, 0.75, 1.0, 1.25, 1.5]
pendulum_run_list = [
    ['base', []],
    ['mass05', make_set_mass(mass_list[0])],
    ['mass075', make_set_mass(mass_list[1])],
    ['mass10', make_set_mass(mass_list[2])],
    ['mass125', make_set_mass(mass_list[3])],
    ['mass15', make_set_mass(mass_list[4])],
]

hopper_run_list = [
    ['base', []]
]

cheetah_run_list = [
    ['base', []]
]

grid = np.meshgrid(hopper_mass_sweep, hopper_friction_sweep)
for mass, fric in np.vstack((grid[0].ravel(), grid[1].ravel())).T:
    hopper_run_list.append(['m_{}_f_{}'.format(mass, fric), make_set_mass_and_fric(fric, mass, mass_body="torso")])

cheetah_grid = np.meshgrid(cheetah_mass_sweep, cheetah_friction_sweep)
for mass, fric in np.vstack((cheetah_grid[0].ravel(), cheetah_grid[1].ravel())).T:
    cheetah_run_list.append(['m_{}_f_{}'.format(mass, fric), make_set_mass_and_fric(fric, mass, mass_body="torso")])


# for x in np.linspace(1, 15.0, 15):
#     lerrel_run_list.append(['mass_{}'.format(x), make_set_mass(x)])

@ray.remote(memory=1500 * 1024 * 1024)
def run_test(test_name, outdir, output_file_name, num_rollouts,
             rllib_config, checkpoint, env_modifier, render):
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
    # high = np.array([1.0, 90.0, env.max_cart_vel, env.max_pole_vel])
    # env.observation_space = spaces.Box(low=-1 * high, high=high, dtype=env.observation_space.dtype)
    if callable(env_modifier):
        env_modifier(env)
    elif len(env_modifier) > 0:
        setattr(env, env_modifier[0], env_modifier[1])
    rewards, step_num = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                                 state_init, action_init, num_rollouts, render)

    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, test_name),
              'wb') as file:
        np.savetxt(file, rewards, delimiter=', ')

    print('The average reward for task {} is {}'.format(test_name, np.mean(rewards)))
    print('The average step length for task {} is {}'.format(test_name, np.mean(step_num)))

    return np.mean(rewards), np.std(rewards), np.mean(step_num), np.std(step_num)


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir, run_list, render=False):

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

    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, "mean_sweep"),
              'wb') as file:
        np.save(file, np.array(temp_output))

    if 'Hopper' in rllib_config['env']:
        reward_means = np.array(temp_output)[1:, 0].reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))
        output_name = output_file_name + 'rew'
        save_heatmap(reward_means, hopper_mass_sweep, hopper_friction_sweep, outdir, output_name, False)

        step_means = np.array(temp_output)[1:, 2].reshape(len(hopper_mass_sweep), len(hopper_friction_sweep))
        output_name = output_file_name + 'steps'
        save_heatmap(step_means, hopper_mass_sweep, hopper_friction_sweep, outdir, output_name, False)

    elif 'Pendulum' in rllib_config['env']:
        means = np.array(temp_output)[1:, 0]
        with open('{}/{}_{}.png'.format(outdir, output_file_name, "transfer_robustness"), 'wb') as transfer_robustness:
            fig = plt.figure()
            plt.bar(np.arange(len(mass_list)), means)
            plt.title("Pendulum performance across mass values")
            plt.xticks(ticks=np.arange(len(mass_list)), labels=["{:0.2f}".format(x) for x in mass_list])
            plt.xlabel("Mass coef")
            plt.savefig(transfer_robustness)
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

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)

    if rllib_config['env'] == "MALerrelPendulumEnv":
        lerrel_run_list = pendulum_run_list
    elif rllib_config['env'] == "MALerrelHopperEnv":
        lerrel_run_list = hopper_run_list
    elif rllib_config['env'] == "MALerrelHopperEnv":
        lerrel_run_list = cheetah_run_list

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name,
                       os.path.join(args.output_dir, date), run_list=lerrel_run_list, render=args.show_images)