import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.mujoco.hopper import HopperEnv
from gym.spaces import Box, Discrete, Dict
import numpy as np
from os import path
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from visualize.plot_heatmap import hopper_friction_sweep, hopper_mass_sweep

class AdvMAHopper(HopperEnv, MultiAgentEnv):
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
        self.cheating = config["cheating"]
        # whether the adversaries are receiving penalties for being too similar
        self.l2_reward = config['l2_reward']
        self.kl_reward = config['kl_reward']
        self.l2_in_tranche = config['l2_in_tranche']
        self.l2_reward_coeff = config['l2_reward_coeff']
        self.kl_reward_coeff = config['kl_reward_coeff']

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
        self.reward_targets = np.linspace(start=self.low_reward, stop=self.high_reward,
                                     num=self.advs_per_strength)
        # repeat the bins so that we can index the adversaries easily
        self.reward_targets = np.repeat(self.reward_targets, self.num_adv_strengths)

        self.comp_adversaries = []
        for i in range(self.adversary_range):
            curr_tranche = int(i / self.num_adv_strengths)
            low_range = max(curr_tranche * self.num_adv_strengths, i - self.num_adv_strengths)
            high_range = min((curr_tranche + 1) * self.num_adv_strengths, i + self.num_adv_strengths)
            self.comp_adversaries.append([low_range, high_range])

        # used to track the previously observed states to induce a memory
        # TODO(@evinitsky) bad hardcoding
        self.obs_size = 11
        if self.cheating:
            self.obs_size += 2
            self.friction_coef = 1.0
            self.mass_coef = 1.0
        self.num_actions = 3
        if self.concat_actions:
            self.obs_size += self.num_actions
        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)

        # Do the initialization
        super(AdvMAHopper, self).__init__()
        self._adv_f_bname = 'foot'
        bnames = self.model.body_names
        self._adv_bindex = bnames.index(
            self._adv_f_bname)  # Index of the body on which the adversary force will be applied
        dr_mass_bname = 'torso'
        self.dr_bindex = bnames.index(dr_mass_bname)
        self.original_friction = np.array(self.model.geom_friction)
        self.original_mass = self.model.body_mass[self.dr_bindex]
        obs_space = self.observation_space
        if self.concat_actions:
            action_space = self.action_space
            low = np.tile(np.concatenate((obs_space.low, action_space.low)), self.num_concat_states)
            high = np.tile(np.concatenate((obs_space.high, action_space.high)), self.num_concat_states)
        else:
            low = np.tile(obs_space.low, self.num_concat_states)
            high = np.tile(obs_space.high, self.num_concat_states)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    @property
    def adv_action_space(self):
        """ 2D adversarial action. Maximum of self.adversary_strength in each dimension.
        """
        return Box(low=-self.adversary_strength, high=self.adversary_strength, shape=(2,))

    @property
    def adv_observation_space(self):
        if self.kl_reward or self.l2_reward:
            dict_space = Dict({'obs': self.observation_space,
                               'is_active': Box(low=-1.0, high=1.0, shape=(1,), dtype=np.int32)})
            return dict_space
        else:
            return self.observation_space

    def _adv_to_xfrc(self, adv_act):
        self.sim.data.xfrc_applied[self._adv_bindex][0] = adv_act[0]
        self.sim.data.xfrc_applied[self._adv_bindex][2] = adv_act[1]

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

    def select_new_adversary(self):
        if self.adversary_range > 0:
            # the -1 corresponds to not having any adversary on at all
            self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)
    
    def randomize_domain(self):
        self.friction_coef = np.random.choice(hopper_friction_sweep)
        self.mass_coef = np.random.choice(hopper_mass_sweep)

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
            hopper_action = actions['agent']

            if self.adversary_range > 0 and 'adversary{}'.format(self.curr_adversary) in actions.keys():
                adv_action = actions['adversary{}'.format(self.curr_adversary)] * self.strengths[self.curr_adversary]
                self._adv_to_xfrc(adv_action)
        else:
            assert actions in self.action_space
            hopper_action = actions
        
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(hopper_action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(hopper_action).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        # you are allowed to observe the mass and friction coefficients
        if self.cheating:
            ob = np.concatenate((ob, [self.mass_coef, self.friction_coef]))
        done = done or self.step_num > self.horizon

        if self.concat_actions:
            self.update_observed_obs(np.concatenate((ob, hopper_action)))
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
                if self.kl_reward or self.l2_reward:
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
                    adv_reward = [(float(self.step_num) / self.horizon) * self.reward_targets[
                       i] -1 * np.abs((float(self.step_num) / self.horizon) * self.reward_targets[
                       i] - self.total_reward) for i in range(self.adversary_range)]
                else:
                    adv_reward = [-reward for _ in range(self.adversary_range)]

                # to do the kl or l2 reward we have to get actions from all the agents
                if self.kl_reward or self.l2_reward:
                    if self.l2_reward and self.adversary_range > 1:
                        action_list = [actions['adversary{}'.format(i)] for i in range(self.adversary_range)]
                        # row index is the adversary, column index is the adversaries you're diffing against
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
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(i): adv_reward[i] + l2_dists_mean[i] *
                                                                 self.l2_reward_coeff for i in range(self.adversary_range)}
                        reward_dict.update(adv_rew_dict)
                    elif self.kl_reward and self.adversary_range > 1:
                        pass

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
            if self.kl_reward or self.l2_reward:
                is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                curr_obs.update({
                    'adversary{}'.format(i): {"obs": self.observed_states, "is_active": np.array([is_active[i]])}
                    for i in range(self.adversary_range)})
            else:
                curr_obs.update({
                    'adversary{}'.format(self.curr_adversary): self.observed_states
                })

        return curr_obs

def hopper_env_creator(env_config):
    env = AdvMAHopper(env_config)
    return env
