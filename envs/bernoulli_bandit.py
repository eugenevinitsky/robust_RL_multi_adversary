"""Agent wants to navigate to a goal, adversary selects the starting point"""

import sys

try:
    import cv2
except:
    pass
import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.env import MultiAgentEnv

if sys.platform == 'darwin':
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib import patches
        from matplotlib.collections import PatchCollection
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
    except:
        pass

PSEUDORANDOM_TRANSFER = 'pseudorandom'

class BernoulliMultiarmBandit(MultiAgentEnv, gym.Env):

    def __init__(self, config):
        self.config = config
        
        self.num_arms = config["num_arms"]
        self.rollout_num = 0

        # This sets how many adversaries exist at each strength level
        self.num_adv_strengths = config["num_adv_strengths"] # num reward ranges
        # This sets how many adversaries exist per strength level
        self.advs_per_strength = config["advs_per_strength"]
        # whether the adversaries are receiving penalties for being too similar
        self.l2_reward = config['l2_reward'] # l2 on / off
        self.l2_reward_coeff = config['l2_reward_coeff']
        self.l2_in_tranche = config['l2_in_tranche'] # l2 within reward range
        self.l2_memory = config['l2_memory'] # use memory, rather than online
        self.l2_memory_target_coeff = config['l2_memory_target_coeff']

        # This sets whether we should use adversaries across a reward range
        self.reward_range = config["reward_range"]
        # This sets the adversaries low reward range
        self.low_reward = config["low_reward"]
        # This sets wthe adversaries high reward range
        self.high_reward = config["high_reward"]

        # This is how many previous observations we concatenate to get the current observation
        self.num_concat_states = config["num_concat_states"]
        assert self.num_concat_states == 1, "Are you sure you want to concat states?"

        # Whether we should compute the reward as regret
        self.regret = config["regret_formulation"]

        self.obs_size = 1 + self.num_arms

        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)

        self.reward_targets = np.linspace(start=self.low_reward, stop=self.high_reward,
                                          num=self.num_adv_strengths)
        # repeat the bins so that we can index the adversaries easily
        self.reward_targets = np.repeat(self.reward_targets, self.advs_per_strength)

        self.horizon = config["horizon"]
        self.step_num = 0

        self.transfer = None

        self.probabilites = []

        self.adversary_range = self.num_adv_strengths * self.advs_per_strength
        self.curr_adversary = 0
        self.total_rew = 0

        # keep track of the actions that they did take
        self.action_list = []

        # TODO(ev) the adversary strengths and the reward ranges operate backwards.
        # TODO(ev) for strengths, self.advs_per_strength gives us the number of adversaries in a given strength tranche
        # TODO(ev) it does the opposite for rewards. There self.num_adv_strengths gives us the number of adversaries
        # TODO(ev) in a given reward tranche.
        # construct the adversaries that we actually compare against for l2 distance.
        # we only compare with adversaries that are in the same self.num_adv_strengths tranche.
        self.comp_adversaries = []
        for i in range(self.adversary_range):
            curr_tranche = int(i / self.num_adv_strengths)
            low_range = max(curr_tranche * self.num_adv_strengths, i - self.num_adv_strengths)
            high_range = min((curr_tranche + 1) * self.num_adv_strengths, i + self.num_adv_strengths)
            self.comp_adversaries.append([low_range, high_range])

        # instantiate the l2 memory tracker
        if self.adversary_range > 0 and self.l2_memory:
            self.global_l2_memory_array = np.zeros((self.adversary_range, self.adv_action_space.low.shape[0]))
            self.local_l2_memory_array = np.zeros((self.adversary_range, self.adv_action_space.low.shape[0]))
            self.local_num_observed_l2_samples = np.zeros(self.adversary_range)

    def update_global_action_mean(self, mean_array):
        self.global_l2_memory_array = (1 - self.l2_memory_target_coeff) * self.global_l2_memory_array + self.l2_memory_target_coeff * mean_array
        self.local_l2_memory_array = np.zeros((self.adversary_range, self.adv_action_space.low.shape[0]))
        self.local_num_observed_l2_samples = np.zeros(self.adversary_range)

    def get_observed_samples(self):
        return self.local_l2_memory_array, self.local_num_observed_l2_samples

    def update_observed_obs(self, new_obs):
        """Add in the new observations and overwrite the stale ones"""
        self.observed_states = np.roll(self.observed_states, shift=self.obs_size, axis=-1)
        self.observed_states[0: self.obs_size] = new_obs
        return self.observed_states

    @property
    def observation_space(self):
        return Box(low=-np.infty, high=np.infty, shape=((1 + self.num_arms) * self.num_concat_states, )) 

    @property
    def action_space(self):
        return Discrete(self.num_arms)

    @property
    def adv_observation_space(self):
        if self.l2_reward and not self.l2_memory:
            dict_space = Dict({'obs': Box(low=-np.infty, high=np.infty, shape=(1, )),
                               'is_active': Box(low=-1.0, high=1.0, shape=(1,), dtype=np.int32)})
            return dict_space
        else:
            return Box(low=-np.infty, high=np.infty, shape=(1, ))

    @property
    def adv_action_space(self):
        low = np.zeros(self.num_arms)
        high = np.ones(self.num_arms)
        return Box(low=np.array(low), high=np.array(high))

    def step(self, action_dict, custom_strategy=None):
        self.step_num += 1
        if self.step_num == 1:
            if self.transfer:
                prng = np.random.RandomState(self.rollout_num)
                if self.transfer == PSEUDORANDOM_TRANSFER:
                    self.probabilites = prng.uniform(0, 1, self.num_arms)
                else:
                    # breaking an abstration barrier here but yolo
                    self.probabilites = self.transfer[0]

                self.rollout_num += 1
                random_arm_order = prng.permutation(self.num_arms)
            elif self.adversary_range > 0:
                self.probabilites = action_dict['adversary{}'.format(self.curr_adversary)]

                # store this since the adversary won't get a reward until the last step
                if self.l2_reward and not self.l2_memory:
                    self.action_list = [action_dict['adversary{}'.format(i)] for i in range(self.adversary_range)]
                if self.l2_memory and self.l2_reward:
                    self.action_list = [action_dict['adversary{}'.format(self.curr_adversary)]]
                    self.local_l2_memory_array[self.curr_adversary] += action_dict['adversary{}'.format(self.curr_adversary)]
                    
                random_arm_order = np.random.permutation(self.num_arms)
            else:
                self.probabilites = np.random.uniform(low=0, high=1, size=self.num_arms)
                random_arm_order = np.random.permutation(self.num_arms)

            if custom_strategy:
                custom_strategy.reset()

            self.probabilites = self.probabilites[random_arm_order]
            # print(self.probabilites)

        if custom_strategy:
            arm_choice = custom_strategy.get_arm(self, self.step_num)
        else:
            arm_choice = action_dict['agent']

        base_rew = np.random.binomial(n=1, p=self.probabilites[arm_choice])

        # print(arm_choice, base_rew)

        if custom_strategy:
            custom_strategy.add_reward(arm_choice, base_rew)

        if self.regret:
            calc_rew = base_rew - max(self.probabilites)
        else:
            calc_rew = base_rew

        done = self.step_num >= self.horizon

        obs = np.zeros(self.num_arms + 1)
        obs[0] = calc_rew
        obs[arm_choice + 1] = 1
        self.update_observed_obs(obs)

        curr_obs = {'agent': self.observed_states}
        curr_rew = {'agent': calc_rew}
        self.total_rew += calc_rew

        if self.adversary_range > 0:

            # the adversaries get observations on the final steps and on the first step
            if done:
                if self.reward_range:
                    # we don't want to give the adversaries an incentive to end the rollout early
                    # so we make a positively shaped reward that peaks at self.reward_targets[i]
                    adv_reward = [-self.reward_targets[i] - np.abs(self.reward_targets[i] - self.total_rew) for i in range(self.adversary_range)]
                else:
                    adv_reward = [-self.total_rew for _ in range(self.adversary_range)]

                if self.l2_reward:
                    # the adversary only takes actions at the first step so
                    # update the reward dict
                    # row index is the adversary, column index is the adversaries you're diffing against
                    # Also, we only want to do the l2 reward against the adversary in a given reward tranch

                    # If we are using the l2_memory, we diff against the mean vector instead of an action vector
                    if not self.l2_memory:
                        if self.l2_in_tranche:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in
                                  self.action_list[self.comp_adversaries[i][0]: self.comp_adversaries[i][1]]]
                                 for i, action_i in enumerate(self.action_list)])
                        else:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in self.action_list]
                                 for action_i in self.action_list])
                        l2_dists_mean = np.sum(l2_dists, axis=-1)
                    else:
                        if self.l2_in_tranche:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in
                                  self.global_l2_memory_array[self.comp_adversaries[i][0]: self.comp_adversaries[i][1]]]
                                 for i, action_i in enumerate(self.action_list)])
                        else:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in self.global_l2_memory_array]
                                 for action_i in self.action_list])
                        l2_dists_mean = np.sum(l2_dists)

                    if self.l2_memory:
                        curr_obs.update({
                            'adversary{}'.format(self.curr_adversary): np.array([0.0, ])})
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]
                                        + l2_dists_mean * self.l2_reward_coeff}
                        curr_rew.update(adv_rew_dict)
                    else:
                        is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                        curr_obs.update({
                            'adversary{}'.format(i): {"obs": np.array([0.0, ]), "is_active": np.array([is_active[i]])}
                            for i in range(self.adversary_range)})
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(i): adv_reward[i] + l2_dists_mean[i] *
                                                                 self.l2_reward_coeff for i in range(self.adversary_range)}
                        curr_rew.update(adv_rew_dict)
                else:
                    curr_obs.update({
                        'adversary{}'.format(self.curr_adversary): np.array([0.0, ])})
                    adv_rew_dict = {'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]}
                    curr_rew.update(adv_rew_dict)

                # else:
                #     pass
                #     # curr_obs.update({
                #     #     'adversary{}'.format(self.curr_adversary): self.curr_pos
                #     # })
                #     # curr_rew.update({
                #     #     'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]
                #     # })

        info = {'agent': {'agent_reward': calc_rew, 'agent_raw_score': base_rew}}

        done_dict = {'__all__': done}

        return curr_obs, curr_rew, done_dict, info


    def reset(self):
        self.step_num = 0
        self.total_rew = 0
        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)

        curr_obs = {'agent': self.observed_states}
        if self.adversary_range > 0:
            if self.l2_reward and not self.l2_memory:
                is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                curr_obs.update({
                    'adversary{}'.format(i): {"obs": np.array([0.0, ]), "is_active": np.array([is_active[i]])}
                    for i in range(self.adversary_range)})
            else:
                curr_obs.update({
                    'adversary{}'.format(self.curr_adversary): np.array([0.0, ])
                })
                if self.l2_memory:
                    self.local_num_observed_l2_samples[self.curr_adversary] += 1

        return curr_obs

    def select_new_adversary(self):
        if self.adversary_range > 0:
            # the -1 corresponds to not having any adversary on at all
            self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)
