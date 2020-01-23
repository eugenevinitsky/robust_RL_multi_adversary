import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.spaces import Box, Discrete
import numpy as np
from os import path
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class AdvMAHalfCheetahEnv(HalfCheetahEnv, MultiAgentEnv):
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

        # here we note that num_adversaries includes the num adv per strength so if we don't divide by this
        # then we are double counting
        self.strengths = np.linspace(start=0, stop=self.adversary_strength,
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
        self.obs_size = 17
        self.num_actions = 6
        if self.concat_actions:
            self.obs_size += self.num_actions
        self.observed_states = np.zeros(self.obs_size * self.num_concat_states)

        # Do the initialization
        super(AdvMAHalfCheetahEnv, self).__init__()
        self._adv_f_bname = ['bfoot', 'ffoot']
        bnames = self.model.body_names
        self._adv_bindex = [bnames.index(i) for i in self._adv_f_bname]   # Index of the body on which the adversary force will be applied

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
        return Box(low=-1, high=1, shape=(4,))

    @property
    def adv_observation_space(self):
        return self.observation_space

    def _adv_to_xfrc(self, adv_act):        
        self.sim.data.xfrc_applied[self._adv_bindex][0] = adv_act[0]
        self.sim.data.xfrc_applied[self._adv_bindex][2] = adv_act[1]

        self.sim.data.xfrc_applied[self._adv_bindex][0] = adv_act[2]
        self.sim.data.xfrc_applied[self._adv_bindex][2] = adv_act[3]

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

    def update_observed_obs(self, new_obs):
        """Add in the new observations and overwrite the stale ones"""
        original_shape = new_obs.shape[0]
        self.observed_states = np.roll(self.observed_states, shift=original_shape, axis=-1)
        self.observed_states[0: original_shape] = new_obs
        return self.observed_states

    def step(self, actions):
        self.step_num += 1
        if isinstance(actions, dict):
            cheetah_action = actions['agent']

            if self.adversary_range > 0 and 'adversary{}'.format(self.curr_adversary) in actions.keys():
                adv_action = actions['adversary{}'.format(self.curr_adversary)] * self.strengths[self.curr_adversary]
                self._adv_to_xfrc(adv_action)
        else:
            assert actions in self.action_space
            cheetah_action = actions
      
        xposbefore = self.sim.data.qpos[0] #note this is different than lerrel's version
        self.do_simulation(cheetah_action, self.frame_skip)
        xposafter = self.sim.data.qpos[0] #note this is different than lerrel's version
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(cheetah_action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = self.step_num > self.horizon

        if self.concat_actions:
            self.update_observed_obs(np.concatenate((ob, cheetah_action)))
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
    env = AdvMAHalfCheetahEnv(env_config)
    return env
