import argparse
import configparser
from copy import deepcopy
import os

import numpy as np
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.rollout import run_rollout
import errno


def reset_config(rllib_config):
    """Return a copy of the config with the possible transfer parameters set to default values"""
    copy_config = deepcopy(rllib_config)
    copy_config['env_config']['friction'] = False
    copy_config['env_config']['change_colors_mode'] = 'no_change'
    copy_config['env_config']['restrict_goal_region'] = True
    copy_config['env_config']['chase_robot'] = False
    copy_config['env_config']['add_gaussian_noise_state'] = False
    copy_config['env_config']['add_gaussian_noise_action'] = False
    return copy_config


@ray.remote
def run_test(test_name, is_env_config, config_value, params_name, params_value, outdir, output_file_name,
             save_trajectory, show_images, num_rollouts,
             rllib_config, checkpoint):
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
    temp_config = reset_config(rllib_config)
    if is_env_config:
        params_string = temp_config['env_config']['env_params']
        params_parser = configparser.RawConfigParser()
        params_parser.read_string(params_string)
        params_parser['sim'][params_name] = str(params_value)
        with open('tmp.config', 'w') as file:
            params_parser.write(file)
        with open('tmp.config', 'r') as file:
            env_params = file.read()
        os.remove('tmp.config')
        temp_config['env_config']['env_params'] = env_params
    # the base case is the only one we don't write update an env param for
    if 'test_name' != 'base' and not is_env_config:
        temp_config['env_config'][test_name] = config_value

    rewards, steps = run_rollout(temp_config, checkpoint, save_trajectory=save_trajectory,
                                               video_file=test_name,
                                               show_images=show_images, num_rollouts=num_rollouts)

    with open(os.path.join(file_path, '{}/{}_{}_rew.txt'.format(outdir, output_file_name, test_name)),
              'wb') as file:
        np.savetxt(file, rewards, delimiter=', ')
    with open(os.path.join(file_path, '{}/{}_{}_steps.txt'.format(outdir, output_file_name, test_name)),
              'wb') as file:
        np.savetxt(file, steps, delimiter=', ')

    print('The average {} reward, episode length is {}, {}'.format(test_name, np.mean(rewards), np.mean(steps)))


def run_transfer_tests(rllib_config, checkpoint, num_rollouts, output_file_name, outdir, save_trajectory, show_images=False):

    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(file_path, outdir)
    if not os.path.exists(output_file_path):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    # test name, is_env_config, config_value, params_name, params_value
    run_list = [
        ['base', False, None, None, None],
        ['friction', False, True, None, None],
        ['change_colors_mode', False, 'ever_step', None, None],
        ['restrict_goal_region', True, False, 'randomize_goals', True],
        ['chase_robot', False, True, None, None],
        ['add_gaussian_noise_state', False, True, None, None],
        ['add_gaussian_noise_action', False, True, None, None]
    ]

    temp_output = [run_test.remote(test_name=list[0], is_env_config=list[1], config_value=list[2],
                 params_name=list[3], params_value=list[4],
                 outdir=outdir, output_file_name=output_file_name,
                 save_trajectory=save_trajectory, show_images=show_images, num_rollouts=num_rollouts,
                 rllib_config=rllib_config, checkpoint=checkpoint) for list in run_list]
    ray.get(temp_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--output_dir', type=str, default='transfer_results',
                        help='')

    parser = replay_parser(parser)
    args = parser.parse_args()
    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_transfer_tests(rllib_config, checkpoint, args.num_rollouts, args.output_file_name, args.output_dir,
                       save_trajectory=args.save_video, show_images=args.show_images)