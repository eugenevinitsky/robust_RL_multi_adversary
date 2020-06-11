import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.spaces import Box, Discrete, Dict
import numpy as np
from os import path
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from visualize.plot_heatmap import cheetah_friction_sweep, cheetah_mass_sweep
from copy import deepcopy
class AdvMAHalfCheetahEnv(HalfCheetahEnv, MultiAgentEnv):
    def __init__(self, config):
        self.horizon = 1000
        self.step_num = 0

        self.total_reward = 0

        self.num_adv_strengths = config["num_adv_strengths"]
        self.adversary_strength = config["adversary_strength"]
        # This sets how many adversaries exist per strength level
        self.advs_per_strength = config["advs_per_strength"]

        # This sets whether we should use adversaries across a reward range
        self.reward_range = config["reward_range"]
        # This sets the adversaries low reward range
        self.low_reward = config["low_reward"]
        # This sets wthe adversaries high reward range
        self.high_reward = config["high_reward"]

        # How frequently we check whether to increase the adversary range
        self.adv_incr_freq = config["adv_incr_freq"]
        # This checks whether we should have a curriculum at all
        self.curriculum = config["curriculum"]
        # The score we use for checking if it is time to increase the number of adversaries
        self.goal_score = config["goal_score"]
        # This is how many previous observations we concatenate to get the current observation
        self.num_concat_states = config["num_concat_states"]
        # This is whether we concatenate the agent action into the observation
        self.concat_actions = config["concat_actions"]
        # This is whether we concatenate the agent action into the observation
        self.domain_randomization = config["domain_randomization"]
        self.extreme_domain_randomization = config["extreme_domain_randomization"]

        self.cheating = config["cheating"]
        # whether the adversaries are receiving penalties for being too similar
        self.l2_reward = config['l2_reward']
        self.kl_reward = config['kl_reward']
        self.l2_in_tranche = config['l2_in_tranche']
        self.l2_memory = config['l2_memory']
        self.l2_memory_target_coeff = config['l2_memory_target_coeff']
        self.l2_reward_coeff = config['l2_reward_coeff']
        self.kl_reward_coeff = config['kl_reward_coeff']
        self.no_end_if_fall = config['no_end_if_fall']
        self.adv_all_actions = config['adv_all_actions']
        self.clip_actions = config['clip_actions']

        # here we note that num_adversaries includes the num adv per strength so if we don't divide by this
        # then we are double counting
        self.strengths = np.linspace(start=0, stop=1,
                                     num=self.num_adv_strengths + 1)[1:]
        # repeat the bins so that we can index the adversaries easily
        self.strengths = np.repeat(self.strengths, self.advs_per_strength)

        # index we use to track how many iterations we have maintained above the goal score
        self.num_iters_above_goal_score = 0

        # This tracks how many adversaries are turned on
        if self.curriculum:
            self.adversary_range = 0
        else:
            self.adversary_range = self.num_adv_strengths * self.advs_per_strength
        if self.adversary_range > 0:
            self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)
        else:
            self.curr_adversary = 0

        # Every adversary at a strength level has different targets. This spurs them to
        # pursue different strategies
        self.num_adv_rews = config['num_adv_rews']
        self.advs_per_rew = config['advs_per_rew']
        self.reward_targets = np.linspace(start=self.low_reward, stop=self.high_reward,
                                     num=self.num_adv_rews)
        # repeat the bins so that we can index the adversaries easily
        self.reward_targets = np.repeat(self.reward_targets, self.advs_per_rew)

        self.comp_adversaries = []
        for i in range(self.adversary_range):
            curr_tranche = int(i / self.advs_per_rew)
            low_range = max(curr_tranche * self.advs_per_rew, i - self.advs_per_rew)
            high_range = min((curr_tranche + 1) * self.advs_per_rew, i + self.advs_per_rew)
            self.comp_adversaries.append([low_range, high_range])

        # used to track the previously observed states to induce a memory
        self.obs_size = 17
        if self.cheating:
            self.obs_size += 2
            self.friction_coef = 1.0
            self.mass_coef = 1.0
        self.num_actions = 6
        if self.concat_actions:
            self.obs_size += self.num_actions
        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)

        # Do the initialization
        super(AdvMAHalfCheetahEnv, self).__init__()
        self._adv_f_bname = ['bfoot', 'ffoot']
        bnames = self.model.body_names
        self._adv_bindex = [bnames.index(i) for i in self._adv_f_bname]   # Index of the body on which the adversary force will be applied
        dr_mass_bname = 'torso' #TODO: check this
        self.dr_bindex = bnames.index(dr_mass_bname)
        self.original_friction = deepcopy(np.array(self.model.geom_friction))
        self.original_mass = deepcopy(self.model.body_mass[self.dr_bindex])
        self.original_mass_all = deepcopy(self.model.body_mass)
        obs_space = self.observation_space
        if self.concat_actions:
            action_space = self.action_space
            low = np.tile(np.concatenate((obs_space.low, action_space.low * 1000)), self.num_concat_states)
            high = np.tile(np.concatenate((obs_space.high, action_space.high * 1000)), self.num_concat_states)
        else:
            low = np.tile(obs_space.low, self.num_concat_states)
            high = np.tile(obs_space.high, self.num_concat_states)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # instantiate the l2 memory tracker
        if self.adversary_range > 0 and self.l2_memory:
            self.global_l2_memory_array = np.zeros(
                (self.adversary_range, self.adv_action_space.low.shape[0], self.horizon + 1))
            self.local_l2_memory_array = np.zeros(
                (self.adversary_range, self.adv_action_space.low.shape[0], self.horizon + 1))
            self.local_num_observed_l2_samples = np.zeros(self.adversary_range)

    @property
    def adv_action_space(self):
        """ 2D adversarial action. Maximum of self.adversary_strength in each dimension.
        """
        if self.adv_all_actions:
            low = np.array(self.action_space.low.tolist())
            high = np.array(self.action_space.high.tolist())
            box = Box(low=-np.ones(low.shape) * self.adversary_strength, high=np.ones(high.shape) * self.adversary_strength,
                      shape=None, dtype=np.float32)
            return box
        else:
            return Box(low=-self.adversary_strength, high=self.adversary_strength, shape=(4,))

    @property
    def adv_observation_space(self):
        if self.kl_reward or (self.l2_reward and not self.l2_memory):
            dict_space = Dict({'obs': self.observation_space,
                               'is_active': Box(low=-1.0, high=1.0, shape=(1,), dtype=np.int32)})
            return dict_space
        else:
            return self.observation_space

    def _adv_to_xfrc(self, adv_act):      
        self.sim.data.xfrc_applied[self._adv_bindex[0]][0] = adv_act[0]
        self.sim.data.xfrc_applied[self._adv_bindex[0]][2] = adv_act[1]

        self.sim.data.xfrc_applied[self._adv_bindex[1]][0] = adv_act[2]
        self.sim.data.xfrc_applied[self._adv_bindex[1]][2] = adv_act[3]

    def update_curriculum(self, mean_rew):
        self.mean_rew = mean_rew
        if self.curriculum:
            if self.mean_rew > self.goal_score:
                self.num_iters_above_goal_score += 1
            else:
                self.num_iters_above_goal_score = 0
            if self.num_iters_above_goal_score >= self.adv_incr_freq:
                self.num_iters_above_goal_score = 0
                self.adversary_range += 1
                self.adversary_range = min(self.adversary_range, self.num_adv_strengths * self.advs_per_strength)

    def get_observed_samples(self):
        return self.local_l2_memory_array, self.local_num_observed_l2_samples

    def update_global_action_mean(self, mean_array):
        """Use polyak averaging to generate an estimate of the current mean actions at each time step"""
        self.global_l2_memory_array = (1 - self.l2_memory_target_coeff) * self.global_l2_memory_array + self.l2_memory_target_coeff * mean_array
        self.local_l2_memory_array = np.zeros(self.local_l2_memory_array.shape)
        self.local_num_observed_l2_samples = np.zeros(self.adversary_range)

    def select_new_adversary(self):
        if self.adversary_range > 0:
            # the -1 corresponds to not having any adversary on at all
            self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)

    def extreme_randomize_domain(self):
        num_geoms = len(self.model.geom_friction)
        num_masses = len(self.model.body_mass)

        self.friction_coef = np.random.choice(cheetah_friction_sweep, num_geoms)[:, np.newaxis]
        self.mass_coef = np.random.choice(cheetah_mass_sweep, num_masses)

        self.model.body_mass[:] = (self.original_mass_all * self.mass_coef)
        self.model.geom_friction[:] = (self.original_friction * self.friction_coef)

    def randomize_domain(self):
        self.friction_coef = np.random.choice(cheetah_friction_sweep)
        self.mass_coef = np.random.choice(cheetah_mass_sweep)

        self.model.body_mass[self.dr_bindex] = (self.original_mass * self.mass_coef)
        self.model.geom_friction[:] = (self.original_friction * self.friction_coef)[:]

    def update_observed_obs(self, new_obs):
        """Add in the new observations and overwrite the stale ones"""
        original_shape = new_obs.shape[0]
        self.observed_states = np.roll(self.observed_states, shift=original_shape, axis=-1)
        self.observed_states[0: original_shape] = new_obs
        return self.observed_states

    def step(self, actions):
        self.step_num += 1
        if isinstance(actions, dict):
            obs_cheetah_action = actions['agent']
            cheetah_action = actions['agent']

            if self.adversary_range > 0 and 'adversary{}'.format(self.curr_adversary) in actions.keys():
                if self.adv_all_actions:
                    adv_action = actions['adversary{}'.format(self.curr_adversary)] * self.strengths[self.curr_adversary]

                    # self._adv_to_xfrc(adv_action)
                    cheetah_action += adv_action
                    # apply clipping to cheetah action
                    if self.clip_actions:
                        cheetah_action = np.clip(obs_cheetah_action, a_min=self.action_space.low, a_max=self.action_space.high)
                else:

                    adv_action = actions['adversary{}'.format(self.curr_adversary)] * self.strengths[self.curr_adversary]
                    self._adv_to_xfrc(adv_action)

        else:
            assert actions in self.action_space
            obs_cheetah_action = actions
            cheetah_action = actions

        # keep track of the action that was taken
        if self.l2_memory and self.l2_reward and isinstance(actions, dict) and 'adversary{}'.format(
                self.curr_adversary) in actions.keys():
            self.local_l2_memory_array[self.curr_adversary, :, self.step_num] += actions[
                'adversary{}'.format(self.curr_adversary)]

        xposbefore = self.sim.data.qpos[0] # note this is different than the RARL version
        self.do_simulation(cheetah_action, self.frame_skip)
        xposafter = self.sim.data.qpos[0] # note this is different than the RARL version
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(cheetah_action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = self.step_num > self.horizon
        # you are allowed to observe the mass and friction coefficients
        if self.cheating:
            ob = np.concatenate((ob, [self.mass_coef, self.friction_coef]))


        if self.concat_actions:
            self.update_observed_obs(np.concatenate((ob, cheetah_action)))
        else:
            self.update_observed_obs(ob)

        self.total_reward += reward
        if isinstance(actions, dict):
            info = {'agent': {'agent_reward': reward}}
            obs_dict = {'agent': self.observed_states}
            reward_dict = {'agent': reward}

            if self.adversary_range > 0 and self.curr_adversary >= 0:
                # to do the kl or l2 reward we have to get actions from all the agents and so we need
                # to pass them all obs
                if self.kl_reward or (self.l2_reward and not self.l2_memory):
                    is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                    obs_dict.update({
                        'adversary{}'.format(i): {"obs": self.observed_states, "is_active": np.array([is_active[i]])}
                        for i in range(self.adversary_range)})
                else:
                    obs_dict.update({
                        'adversary{}'.format(self.curr_adversary): self.observed_states
                    })

                if self.reward_range:
                    # we make this a positive reward that peaks at the reward target so that the adversary
                    # isn't trying to make the rollout end as fast as possible. It wants the rollout to continue.

                    # we also rescale by horizon because this can BLOW UP

                    # an explainer because this is confusing. We are trying to get the agent to a reward target.
                    # we treat the reward as evenly distributed per timestep, so at each time we take the abs difference
                    # between a linear function of step_num from 0 to the target and the current total reward.
                    # we then subtract this value off from the linear function again. This creates a reward
                    # that peaks at the target value. We then scale it by (1 / max(1, self.step_num)) because
                    # if we are not actually able to hit the target, this reward can blow up.
                    adv_reward = [((float(self.step_num) / self.horizon) * self.reward_targets[
                       i] - 1 * np.abs((float(self.step_num) / self.horizon) * self.reward_targets[
                       i] - self.total_reward)) * (1 / max(1, self.step_num)) for i in range(self.adversary_range)]
                else:
                    adv_reward = [-reward for _ in range(self.adversary_range)]

                if self.l2_reward and self.adversary_range > 1:
                    # to do the kl or l2 reward exactly we have to get actions from all the agents
                    if self.l2_reward and not self.l2_memory:
                        action_list = [actions['adversary{}'.format(i)] for i in range(self.adversary_range)]
                        # only diff against agents that have the same reward goal
                        if self.l2_in_tranche:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in
                                  action_list[self.comp_adversaries[i][0]: self.comp_adversaries[i][1]]]
                                 for i, action_i in enumerate(action_list)])
                        else:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in action_list]
                                 for action_i in action_list])
                        # This matrix is symmetric so it shouldn't matter if we sum across rows or columns.
                        l2_dists_mean = np.sum(l2_dists, axis=-1)
                    # here we approximate the l2 reward by diffing against the average action other agents took
                    # at this timestep
                    if self.l2_reward and self.l2_memory:
                        action_list = [actions['adversary{}'.format(self.curr_adversary)]]
                        if self.l2_in_tranche:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in
                                  self.global_l2_memory_array[self.comp_adversaries[i][0]: self.comp_adversaries[i][1], :, self.step_num]]
                                 for i, action_i in enumerate(action_list)])
                        else:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in self.global_l2_memory_array[:, :, self.step_num]]
                                 for action_i in action_list])
                        l2_dists_mean = np.sum(l2_dists)

                    if self.l2_memory:
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]
                                                                                   + l2_dists_mean * self.l2_reward_coeff}
                    else:
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(i): adv_reward[i] + l2_dists_mean[i] *
                                                                 self.l2_reward_coeff for i in
                                        range(self.adversary_range)}
                    reward_dict.update(adv_rew_dict)


                else:
                    reward_dict.update({'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]})

            done_dict = {'__all__': done}
            return obs_dict, reward_dict, done_dict, info
        else:
            return ob, reward, done, {}

    def reset(self):
        self.step_num = 0
        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)
        self.total_reward = 0
        obs = super().reset()

        if self.concat_actions:
            self.update_observed_obs(np.concatenate((obs, [0.0] * 3)))
        else:
            self.update_observed_obs(obs)

        curr_obs = {'agent': self.observed_states}
        if self.adversary_range > 0 and self.curr_adversary >= 0:
            if self.kl_reward or (self.l2_reward and not self.l2_memory):
                is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                curr_obs.update({
                    'adversary{}'.format(i): {"obs": self.observed_states, "is_active": np.array([is_active[i]])}
                    for i in range(self.adversary_range)})
            else:
                curr_obs.update({
                    'adversary{}'.format(self.curr_adversary): self.observed_states
                })

            # track how many times each adversary was used
            if self.l2_memory:
                self.local_num_observed_l2_samples[self.curr_adversary] += 1

        return curr_obs

def cheetah_env_creator(env_config):
    env = AdvMAHalfCheetahEnv(env_config)
    return env
