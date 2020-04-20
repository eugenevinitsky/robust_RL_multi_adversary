import argparse
import configparser
from copy import deepcopy
from datetime import datetime
from gym import spaces
import os
import pytz
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import ray

from utils.parsers import replay_parser
from utils.rllib_utils import get_config
from visualize.mujoco.run_rollout import run_rollout, instantiate_rollout
from visualize.plot_heatmap import save_heatmap, hopper_friction_sweep, hopper_mass_sweep, cheetah_friction_sweep, cheetah_mass_sweep
import errno
from visualize.mujoco.transfer_tests import reset_env

def main():
    parser = argparse.ArgumentParser('Parse configuration file')

    parser = replay_parser(parser)
    parser.add_argument('--output_dir', type=str, default='visualize_results',
                        help='Directory to output the files into')
    parser.add_argument('--render',action='store_true', default=False,
                        help='Render env?')
    args = parser.parse_args()

    rllib_config, checkpoint = get_config(args)
    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})

    ray.init(num_cpus=args.num_cpus)

    env, agent, multiagent, use_lstm, policy_agent_mapping, state_init, action_init = instantiate_rollout(rllib_config,
                                                                                                         checkpoint)
    num_adversaries = env.num_adv_strengths * env.advs_per_strength
    obs_per_adv = {}
    for i in range(num_adversaries):
        rewards, step_num, final_obs = run_rollout(env, agent, multiagent, use_lstm, policy_agent_mapping,
                                    state_init, action_init, args.num_rollouts, args.render, adversary=i)
        obs_per_adv['adversary{}'.format(i)] = np.asarray(final_obs)


    colors = list("rgbcmyk")

    for data in obs_per_adv.values():
        x = data[:,0]
        y = data[:,1]
        plt.scatter(x, y, color=colors.pop())

    plt.legend(obs_per_adv.keys())
    plt.title('Average velocities for {} Adversaries in Hopper'.format(num_adversaries,env))
    exp_title = args.result_dir.split('/')

    Path('{}/mujoco/{}/{}'.format(args.output_dir, exp_title[-4], exp_title[-2])).mkdir(parents=True, exist_ok=True)

    plt.savefig('{}/mujoco/{}/{}/{}'.format(args.output_dir, exp_title[-4], exp_title[-2], exp_title[-1] + 'final_velocity'))

if __name__ == '__main__':
    main()