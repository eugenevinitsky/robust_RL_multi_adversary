from os import path
import sys

import gym
from gym.spaces import Box, Tuple, Discrete
from gym import spaces
from gym.utils import seeding
import numpy as np
from ray.rllib.env import MultiAgentEnv

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
        self.gaussian_state_noise_scale = 0.1
        self.gaussian_action_noise_scale = 0.1

        self.viewer = None

        self.high = np.array([1., 1., self.max_speed])
        self.seed()

    @property
    def action_space(self):
        return spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Box(low=-self.high, high=self.high, dtype=np.float32)

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
        if self.add_gaussian_state_noise:
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
        self.model_based = config["model_based"]
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)

    @property
    def adv_action_space(self):
        return Box(low=-3, high=3, shape=(1,))

    @property
    def adv_observation_space(self):
        return self.observation_space

    def select_new_adversary(self):
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)

    def step(self, actions):
        pendulum_action = actions['pendulum']
        if 'adversary{}'.format(self.curr_adversary) in actions.keys():
            adv_action = actions['adversary{}'.format(self.curr_adversary)]
            pendulum_action += adv_action * self.adversary_strength
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


class ModelBasedPendulumEnv(PendulumEnv):
    def __init__(self, config):
        """Env with adversaries that are just sinusoids. Used for testing out identification schemes."""
        super(ModelBasedPendulumEnv, self).__init__()
        self.num_adversaries = config["num_adversaries"]
        self.adversary_strength = config["adversary_strength"]
        self.model_based = config["model_based"]
        self.guess_adv = config['guess_adv']
        # TODO(use an autoregressive model to condition this guess on your action)
        self.guess_next_state = config['guess_next_state']
        self.num_concat_states = config['num_concat_states']
        # used to track the previously observed states
        self.observed_states = np.zeros(self.observation_space.shape[0])
        self.correct_adv_score = 0.4
        self.correct_state_coeff = 1.0
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)
        self.num_correct_guesses = 0
        self.state_error = np.zeros(self.observation_space.shape[0])
        # purely used for readout and visualization
        self.curr_state_error = np.zeros(self.observation_space.shape[0])
        # attribute that can be used to turn the adversary off for testing
        self.has_adversary = True
        # the reward without any auxiliary rewards
        self.true_rew = 0.0

    @property
    def observation_space(self):
        obs_space = super().observation_space
        low = np.repeat(obs_space.low, self.num_concat_states)
        high = np.repeat(obs_space.high, self.num_concat_states)
        return Box(low=low, high=high, dtype=np.float32)

    @property
    def action_space(self):
        if not self.guess_adv and not self.guess_next_state:
            return super().action_space
        elif self.guess_adv and not self.guess_next_state:
            return Tuple((super().action_space, Discrete(self.num_adversaries)))
        elif self.guess_next_state and not self.guess_adv:
            return Tuple((super().action_space, super().observation_space))
        else:
            return Tuple((super().action_space, super().observation_space, Discrete(self.num_adversaries)))

    def step(self, action):
        if not self.guess_adv and not self.guess_next_state:
            torque = action
        elif self.guess_adv and not self.guess_next_state:
            torque = action[0]
            adv_guess = action[1]
        elif self.guess_next_state and not self.guess_adv:
            torque = action[0]
            state_guess = action[1]
        else:
            torque = action[0]
            state_guess = action[1]
            adv_guess = action[2]
        # Sinusoidal perturbation
        if self.has_adversary:
            adv_action = np.sin(2 * np.pi * self.curr_adversary * self.step_num * self.dt)
            torque += adv_action * self.adversary_strength
        if isinstance(self.action_space, Box):
            pendulum_action = np.clip(torque, a_min=self.action_space.low, a_max=self.action_space.high)
        elif isinstance(self.action_space, Tuple):
            pendulum_action = np.clip(torque, a_min=self.action_space[0].low, a_max=self.action_space[0].high)
        else:
            sys.exit('How did you get here my friend. Only Box and Tuple action spaces are handled right now.')
        obs, rew, done, info = super().step(pendulum_action)

        self.true_rew = rew

        if self.guess_adv:
            if int(adv_guess) == self.curr_adversary:
                rew += self.correct_adv_score
                self.num_correct_guesses += 1

        if self.guess_next_state:
            self.curr_state_error = np.abs(state_guess.flatten() - self._get_obs())
            self.state_error += np.abs(state_guess.flatten() - self._get_obs())
            rew -= np.linalg.norm(state_guess - obs) * self.correct_state_coeff

        # TODO add true reward, without auxiliary reward to info
        return self.update_observed_obs(obs), rew, done, info

    def update_observed_obs(self, new_obs):
        original_shape = super().observation_space.shape[0]
        self.observed_states = np.roll(self.observed_states, shift=original_shape, axis=-1)
        self.observed_states[0: original_shape] = new_obs

    def reset(self):
        self.num_correct_guesses = 0
        self.true_rew = 0.0

        self.curr_state_error = np.zeros(self.observation_space.shape[0])
        self.state_error = np.zeros(self.observation_space.shape[0])
        return self.update_observed_obs(super().reset())

    def select_new_adversary(self):
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)