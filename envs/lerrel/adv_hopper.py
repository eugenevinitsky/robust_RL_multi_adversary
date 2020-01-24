import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.mujoco.hopper import HopperEnv
from gym.spaces import Box, Discrete
import numpy as np
from os import path
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from visualize.plot_heatmap import hopper_friction_sweep, hopper_mass_sweep

class AdvMAHopper(HopperEnv, MultiAgentEnv):
    def __init__(self, config):
        self.horizon = 1000
        self.step_num = 0

        self.num_adv_strengths = config["num_adv_strengths"]
        self.adversary_strength = config["adversary_strength"]
        # This sets how many adversaries exist per strength level
        self.advs_per_strength = config["advs_per_strength"]
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
            self.curr_adversary = np.random.randint(low=-1, high=self.adversary_range)
    
    def randomize_domain(self):
        self.friction_coef = np.random.choice(hopper_friction_sweep)
        self.mass_coef = np.random.choice(hopper_mass_sweep)

        mass_bname = 'torso'
        bnames = self.model.body_names
        bindex = bnames.index(mass_bname)
        self.model.body_mass[bindex] = (self.model.body_mass[bindex] * self.mass_coef)
        self.model.geom_friction[:] = (self.model.geom_friction * self.friction_coef)[:]

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

        if isinstance(actions, dict):
            info = {'agent': {'agent_reward': reward}}
            obs_dict = {'agent': self.observed_states}
            reward_dict = {'agent': reward}

            if self.adversary_range > 0 and self.curr_adversary >= 0:
                obs_dict.update({
                    'adversary{}'.format(self.curr_adversary): self.observed_states
                })

                reward_dict.update({'adversary{}'.format(self.curr_adversary): -reward})

            done_dict = {'__all__': done}
            return obs_dict, reward_dict, done_dict, info
        else:
            return ob, reward, done, {}

    def reset(self):
        self.step_num = 0
        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)
        obs = super().reset()

        if self.concat_actions:
            self.update_observed_obs(np.concatenate((obs, [0.0] * 3)))
        else:
            self.update_observed_obs(obs)

        curr_obs = {'agent': self.observed_states}
        if self.adversary_range > 0 and self.curr_adversary >= 0:
            curr_obs.update({'adversary{}'.format(self.curr_adversary):
                                 self.observed_states})

        return curr_obs

def hopper_env_creator(env_config):
    env = AdvMAHopper(env_config)
    return env
