import argparse
import configparser
from copy import deepcopy
import collections
from datetime import datetime
from gym import spaces
import os
import pytz

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import ray
from ray.rllib.evaluation.episode import _flatten_action

from envs.multiarm_bandit import PSEUDORANDOM_TRANSFER
from utils.parsers import replay_parser
from utils.rllib_utils import get_config_from_path
from visualize.mujoco.run_rollout import run_rollout, instantiate_rollout, DefaultMapping
from visualize.plot_heatmap import save_heatmap, hopper_friction_sweep, hopper_mass_sweep, cheetah_friction_sweep, cheetah_mass_sweep
import errno

def run_adv_versus_transfer_tests(rllib_agent_config, checkpoint, rllib_adv_config, adv_checkpoint, adv_num, num_rollouts, output_file_name, outdir, render=False):
    output_file_path = os.path.join(outdir, output_file_name)
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    _, adv_agent, _, _, adv_policy_agent_mapping, _, adv_action_init = instantiate_rollout(rllib_adv_config, adv_checkpoint)
    env, agent, multiagent, _, policy_agent_mapping, _, action_init = instantiate_rollout(rllib_agent_config, checkpoint)

    env.domain_randomization = False
    env.adversary_range = 1

    rewards = []
    step_nums = []

    # actually do the rollout
    for r_itr in range(num_rollouts):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        prev_actions = DefaultMapping(lambda agent_id: action_init[mapping_cache[agent_id]])
        adv_prev_actions = DefaultMapping(lambda agent_id: adv_action_init[mapping_cache[agent_id]])
        obs = env.reset()
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        step_num = 0
        while not done:
            step_num += 1
            agent_id = 'agent'
            agent_obs = obs['agent']
            adversary_id = 'adversary{}'.format(adv_num)
            adversary_obs = np.array([0.0])
            action_dict = {}
            
            agent_policy_id = mapping_cache.setdefault(agent_id, policy_agent_mapping(agent_id))
            agent_prev_action = _flatten_action(prev_actions[agent_id])
            agent_action = agent.compute_action(agent_obs,
                                                prev_action=agent_prev_action,
                                                prev_reward=prev_rewards[agent_id],
                                                policy_id=agent_policy_id)
            action_dict[agent_id] = agent_action
            prev_action = _flatten_action(agent_action)
            prev_actions[agent_id] = prev_action

            adversary_policy_id = mapping_cache.setdefault(adversary_id, adv_policy_agent_mapping(adversary_id))
            adversary_prev_action = _flatten_action(adv_prev_actions[adversary_id])
            adversary_action = adv_agent.compute_action(adversary_obs,
                                                prev_action=adversary_prev_action,
                                                prev_reward=prev_rewards[adversary_id],
                                                policy_id=adversary_policy_id)
            action_dict["adversary0"] = adversary_action
            prev_action = _flatten_action(adversary_action)
            adv_prev_actions[adversary_id] = prev_action
            
            # we turn the adversaries off so you only send in the pendulum keys
            next_obs, reward, done, info = env.step(action_dict)
            if render:
                env.render()
            done = done['__all__']
            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r

            # we only want the robot reward, not the adversary reward
            reward_total += info['agent']['agent_reward']
            obs = next_obs
        print("Episode reward", reward_total)

        rewards.append(reward_total)
        step_nums.append(step_num)

    print('the average reward is ', np.mean(rewards))
    return rewards, step_num

    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, "with_adv_mean_sweep"),
            'wb') as file:
        np.savetxt(file, np.array(temp_output))

    with open('{}/{}_{}'.format(outdir, output_file_name, "adv_scores.png"),
            'wb') as file:
        means = np.array(temp_output)[:,0]
        fig = plt.figure()
        plt.bar(reversed(np.arange(num_advs)), means)
        plt.title("Scores under each adversary")
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
    parser.add_argument('--adv_num', required=True, type=int, help='')


    parser = replay_parser(parser)

    parser.add_argument(
        'adv_checkpoint_dr', type=str, help='Directory containing results')
    parser.add_argument('adv_checkpoint_num', type=str, help='Checkpoint number.')
    args = parser.parse_args()

    print(args.adv_checkpoint_dr)
    print(args.adv_checkpoint_num)

    rllib_config, checkpoint = get_config_from_path(args.result_dir, args.checkpoint_num)
    adv_rllib_config, adv_checkpoint = get_config_from_path(args.adv_checkpoint_dr, args.adv_checkpoint_num)

    ray.init(num_cpus=args.num_cpus)

    if 'run' not in rllib_config['env_config']:
        rllib_config['env_config'].update({'run': 'PPO'})
    run_adv_versus_transfer_tests(rllib_config, checkpoint, adv_rllib_config, adv_checkpoint, args.adv_num, args.num_rollouts,
                                  args.output_file_name, os.path.join(args.output_dir, date), render=args.show_images)