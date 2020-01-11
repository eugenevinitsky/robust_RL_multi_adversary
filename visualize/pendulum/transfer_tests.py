import argparse
import configparser
from copy import deepcopy
import os

import numpy as np
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.pendulum.run_rollout import run_rollout, instantiate_rollout
import errno


def make_set_friction(friction_coef):
    def set_friction(env):
        # nonlocal friction_coef
        env.model.geom_friction[:] = (env.model.geom_friction * friction_coef)[:]
    return set_friction

def make_set_mass(mass_coef, mass_body='pole'):
    def set_mass(env):
        # nonlocal friction_coef
        env.model.body_mass[2] = (env.model.body_mass[2] * mass_coef)
    return set_mass
    

# test name, is_env_config, config_value, params_name, params_value
run_list = [
    ['base', []],
    ['friction', ['friction', True]],
    ['gaussian_action_noise', ['add_gaussian_action_noise', True]],
    ['gaussian_state_noise', ['add_gaussian_state_noise', True]]
]

# test name, is_env_config, config_value, params_name, params_value
lerrel_run_list = [
    # ['base', []],
    # ['friction', make_set_friction(0.2)],
    ['mass', make_set_mass(0.5)],
    # ['gaussian_state_noise', ['add_gaussian_state_noise', True]]
]


@ray.remote
def run_test(test_name, outdir, output_file_name, num_rollouts,
             rllib_config, checkpoint, env_modifier):
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
    file_path = os.path.dirname(os.path.abspath(__file__))
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
    if callable(env_modifier):
            env_modifier(env)
    elif len(env_modifier) > 0:
        setattr(env, env_modifier[0], env_modifier[1])
    rewards = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                                 state_init, action_init, num_rollouts)

    with open(os.path.join(file_path, '{}/{}_{}_rew.txt'.format(outdir, output_file_name, test_name)),
              'wb') as file:
        np.savetxt(file, rewards, delimiter=', ')

    print('The average reward for task {} is {}'.format(test_name, np.mean(rewards)))


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir):

    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    temp_output = [run_test.remote(test_name=list[0],
                 outdir=outdir, output_file_name=output_file_name,
                 num_rollouts=num_rollouts,
                 rllib_config=rllib_config, checkpoint=checkpoint, env_modifier=list[1]) for list in lerrel_run_list]
    ray.get(temp_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out/pendulum',
                        help='The file name we use to save our results')
    parser.add_argument('--output_dir', type=str, default='transfer_results',
                        help='')

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name, args.output_dir)