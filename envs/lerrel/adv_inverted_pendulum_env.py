import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.spaces import Box, Discrete
import numpy as np
from os import path
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class AdvMAPendulumEnv(InvertedPendulumEnv, MultiAgentEnv):
    def __init__(self, config):
        self.num_adversaries = config["num_adversaries"]
        self.adversary_strength = config["adversary_strength"]
        self.max_cart_vel = 100
        self.max_pole_vel = 100
        self.horizon = 1000
        self.step_num = 0
        self.select_new_adversary()
        super(AdvMAPendulumEnv, self).__init__()
        self._adv_f_bname = 'pole'
        bnames = self.model.body_names
        self._adv_bindex = bnames.index(self._adv_f_bname) #Index of the body on which the adversary force will be applied

        high = np.array([1.0, 90.0, self.max_cart_vel, self.max_pole_vel])
        self.observation_space = spaces.Box(low=-1 * high, high=high, dtype=self.observation_space.dtype)
        # TODO(kp): find a better obs norm
        self.obs_norm = 1.0

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

    def select_new_adversary(self):
        if self.num_adversaries:
            self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)

    def step(self, actions):
        self.step_num += 1
        if isinstance(actions, dict):
            pendulum_action = actions['pendulum']

            if self.num_adversaries and 'adversary{}'.format(self.curr_adversary) in actions.keys():
                adv_action = actions['adversary{}'.format(self.curr_adversary)]
                self._adv_to_xfrc(adv_action)
        else:
            assert actions in self.action_space
            pendulum_action = actions
        
        self.sim.data.qvel[0] = np.clip(self.sim.data.qvel[0], -self.max_cart_vel, self.max_cart_vel)
        self.sim.data.qvel[1] = np.clip(self.sim.data.qvel[1], -self.max_pole_vel, self.max_pole_vel)

        reward = 1.0
        self.do_simulation(pendulum_action, self.frame_skip)
        ob = self._get_obs()
        done = not np.isfinite(ob).all() or np.abs(ob[1]) > .2 or self.step_num > self.horizon
        
        if isinstance(actions, dict):
            info = {'pendulum': {'pendulum_reward': reward}}
            obs_dict = {'pendulum': ob}
            reward_dict = {'pendulum': reward}

            for i in range(self.num_adversaries):
                # is_active = 1 if self.curr_adversary == i else 0
                obs_dict.update({
                    'adversary{}'.format(i): ob
                })

                reward_dict.update({'adversary{}'.format(i): -reward})

            done_dict = {'__all__': done}
            return obs_dict, reward_dict, done_dict, info
        else:
            return ob, reward, done, {}     


    def reset(self):
        self.step_num = 0
        obs = super().reset()
        curr_obs = {'pendulum': obs}
        for i in range(self.num_adversaries):
            is_active = 1 if self.curr_adversary == i else 0
            curr_obs.update({'adversary{}'.format(i):
                                 obs})

        return curr_obs
