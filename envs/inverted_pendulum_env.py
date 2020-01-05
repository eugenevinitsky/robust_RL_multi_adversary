from gym.spaces import Box, Discrete

from ray.rllib.env import MultiAgentEnv

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0, horizon=200):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.horizon = horizon
        self.step_num = 0
        self.obs_norm = 50.0
        self.should_render = False
        self.friction = False
        self.friction_coef = 0.2
        self.add_gaussian_state_noise = False
        self.add_gaussian_action_noise = False
        self.gaussian_state_noise_scale = 0.2
        self.gaussian_action_noise_scale = 0.2

        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        self.step_num += 1
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        if self.add_gaussian_action_noise:
            u = np.clip(u + self.gaussian_action_noise_scale * np.random.normal(), -self.max_torque, self.max_torque)
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        if self.friction:
            newthdot -= self.friction_coef * thdot
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        done = False
        if self.step_num > self.horizon:
            done = True

        if self.should_render:
            self.render()
        obs = self._get_obs() / self.obs_norm
        if self.add_gaussian_action_noise:
            obs += self.gaussian_state_noise_scale * np.random.normal(size=obs.shape)
        return obs, -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.step_num = 0
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs() / self.obs_norm

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


class MAPendulumEnv(PendulumEnv, MultiAgentEnv):
    def __init__(self, config):
        super(MAPendulumEnv, self).__init__()
        self.num_adversaries = config["num_adversaries"]
        self.adversary_strength = config["adversary_strength"]
        self.adversary_action_dim_size = config["adversary_action_dim"]
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)
        self.adv_actions = np.linspace(-3, 3, self.adversary_action_dim_size)

    @property
    def adv_action_space(self):
        return Discrete(self.adversary_action_dim_size)

    @property
    def adv_observation_space(self):
        return self.observation_space

    def select_new_adversary(self):
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)

    def step(self, actions):
        pendulum_action = actions['pendulum']
        if 'adversary{}'.format(self.curr_adversary) in actions.keys():
            adv_action = actions['adversary{}'.format(self.curr_adversary)]
            pendulum_action += self.adv_actions[adv_action] * self.adversary_strength
            pendulum_action = np.clip(pendulum_action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs, reward, done, info = super().step(pendulum_action)
        info = {'pendulum': {'pendulum_reward': reward}}
        obs_dict = {'pendulum': obs, 'adversary{}'.format(self.curr_adversary): obs}
        reward_dict = {'pendulum': reward, 'adversary{}'.format(self.curr_adversary): -reward}
        done_dict = {'__all__': done}
        return obs_dict, reward_dict, done_dict, info

    def reset(self):
        obs = super().reset()
        return {'pendulum': obs, 'adversary{}'.format(self.curr_adversary): obs}