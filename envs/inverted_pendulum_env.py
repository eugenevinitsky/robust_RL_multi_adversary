from gym.spaces import Box

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

    def get_observation_space(self):
        return self.observation_space

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
        # If this is true we are doing a custom training run that requires all of the adversaries to get
        # an observation at every timestep
        self.kl_diff_training = config["kl_diff_training"]
        # This sets how many adversaries exist per strength level
        self.num_adv_per_strength = config["num_adv_per_strength"]
        # How frequently we check whether to increase the adversary range
        self.adv_incr_freq = config["adv_incr_freq"]
        # This checks whether we should have a curriculum at all
        self.curriculum = config["curriculum"]
        # The score we use for checking if it is time to increase the number of adversaries
        self.goal_score = config["goal_score"]
        self.mean_rew = 0.0
        # index we use to track how many iterations we have maintained above the goal score
        self.num_iters_above_goal_score = 0

        # here we note that num_adversaries includes the num adv per strength so if we don't divide by this
        # then we are double counting
        self.strengths = np.linspace(start=0, stop=self.adversary_strength,
                                     num=(self.num_adversaries / self.num_adv_per_strength) + 1)[1:]
        # repeat the bins so that we can index the adversaries easily
        self.strengths = np.repeat(self.strengths, self.num_adv_per_strength)
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)
        # This tracks how many adversaries are turned on
        if self.curriculum:
            self.adversary_range = 0
        else:
            self.adversary_range = self.num_adversaries

    @property
    def adv_action_space(self):
        return Box(low=-3, high=3, shape=(1,))

    @property
    def adv_observation_space(self):
        if self.kl_diff_training:
            dict_space = spaces.Dict({'obs': super().get_observation_space(),
                                      'is_active': Box(low=-1.0, high=1.0, shape=(1,), dtype=np.int32)})
            return dict_space
        else:
            return self.observation_space

    def update_curriculum(self, mean_rew):
        self.mean_rew = mean_rew
        if self.curriculum:
            if self.mean_rew > self.goal_score:
                self.num_iters_above_goal_score += 1
            else:
                self.num_iters_above_goal_score = 0
            if self.num_iters_above_goal_score >= self.adv_incr_freq:
                self.num_iters_above_goal_score = 0
                self.adversary_range += self.num_adv_per_strength
                self.adversary_range = min(self.adversary_range, self.num_adversaries)

    def select_new_adversary(self):
        if self.adversary_range > 0:
            self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)

    def step(self, actions):
        pendulum_action = actions['pendulum']
        if 'adversary{}'.format(self.curr_adversary) in actions.keys():
            adv_action = actions['adversary{}'.format(self.curr_adversary)]
            pendulum_action += adv_action * self.strengths[self.curr_adversary]
            pendulum_action = np.clip(pendulum_action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs, reward, done, info = super().step(pendulum_action)
        info = {'pendulum': {'pendulum_reward': reward}}

        obs_dict = {'pendulum': obs}
        reward_dict = {'pendulum': reward}

        if self.kl_diff_training:
            for i in range(self.num_adversaries):
                is_active = 1 if self.curr_adversary == i else 0
                obs_dict.update({
                    'adversary{}'.format(i): {'obs': obs, 'is_active': np.array([is_active])}
                })

                reward_dict.update({'adversary{}'.format(i): -reward})
        else:
            if self.adversary_range > 0:
                obs_dict.update({'adversary{}'.format(self.curr_adversary): obs})
                reward_dict.update({'adversary{}'.format(self.curr_adversary): -reward})

        done_dict = {'__all__': done}
        return obs_dict, reward_dict, done_dict, info

    def reset(self):
        obs = super().reset()
        curr_obs = {'pendulum': obs}
        if self.kl_diff_training:
            for i in range(self.num_adversaries):
                is_active = 1 if self.curr_adversary == i else 0
                curr_obs.update({'adversary{}'.format(i):
                                     {'obs': obs,
                                      'is_active': np.array([is_active])
                                     }})
        else:
            if self.adversary_range > 0:
                curr_obs.update({'adversary{}'.format(self.curr_adversary): obs})

        return curr_obs