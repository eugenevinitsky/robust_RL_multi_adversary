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

    def __init__(self, config, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.horizon = config["horizon"]
        self.step_num = 0
        self.should_render = False
        self.friction = False
        self.friction_coef = 0.1
        self.add_gaussian_state_noise = False
        self.add_gaussian_action_noise = False
        self.gaussian_state_noise_scale = 0.1
        self.gaussian_action_noise_scale = 0.1

        self.viewer = None

        self.high = np.array([1., 1., self.max_speed])
        self.low = np.array([-1., -1., -self.max_speed])
        self.obs_norm = self.high * 2
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

    def step(self, u):
        """state_perturbation is a direct perturbation of the theta_dot update"""
        self.step_num += 1
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        if self.add_gaussian_action_noise:
            newthdot += self.gaussian_action_noise_scale * np.random.normal()

        if self.friction:
            newthdot -= self.friction_coef * thdot * dt
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
        obs = np.clip(obs, a_min=self.low, a_max=self.high)
        return obs, -costs, done, {}

    def get_reward(self, state, action):
        th, thdot = state
        return angle_normalize(th)**2 + .1*thdot**2 + .001*(action**2)

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
        super(MAPendulumEnv, self).__init__(config)
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
        # This is how many previous observations we concatenate to get the current observation
        self.num_concat_states = config["num_concat_states"]
        # This is whether we concatenate the agent action into the observation
        self.concat_actions = config["concat_actions"]
        # used to track the previously observed states
        self.observed_states = np.zeros(self.observation_space.shape[0])

        self.mean_rew = 0.0
        # index we use to track how many iterations we have maintained above the goal score
        self.num_iters_above_goal_score = 0

        # here we note that num_adversaries includes the num adv per strength so if we don't divide by this
        # then we are double counting
        self.strengths = np.linspace(start=0, stop=self.adversary_strength,
                                     num=self.num_adversaries + 1)[1:]
        # repeat the bins so that we can index the adversaries easily
        self.strengths = np.repeat(self.strengths, self.num_adv_per_strength)
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)
        # This tracks how many adversaries are turned on
        if self.curriculum:
            self.adversary_range = 0
        else:
            self.adversary_range = self.num_adversaries * self.num_adv_per_strength

    @property
    def observation_space(self):
        obs_space = super().observation_space
        if self.concat_actions:
            action_space = super().action_space
            low = np.tile(np.concatenate((obs_space.low, action_space.low)), self.num_concat_states)
            high = np.tile(np.concatenate((obs_space.high, action_space.high)), self.num_concat_states)
        else:
            low = np.tile(obs_space.low, self.num_concat_states)
            high = np.tile(obs_space.high, self.num_concat_states)
        return Box(low=low, high=high, dtype=np.float32)

    @property
    def adv_action_space(self):
        return Box(low=-3, high=3, shape=(1,))

    @property
    def adv_observation_space(self):
        if self.kl_diff_training:
            # TODO(ydu, evinitsky) should we let the adversaries see the history too?
            # for now I'm letting them see it, but maybe that makes them too powerful
            dict_space = spaces.Dict({'obs': self.observation_space,
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

    def update_observed_obs(self, new_obs):
        """Add in the new observations and overwrite the stale ones"""
        original_shape = int(self.observation_space.shape[0] / self.num_concat_states)
        self.observed_states = np.roll(self.observed_states, shift=original_shape, axis=-1)
        self.observed_states[0: original_shape] = new_obs
        return self.observed_states

    def step(self, actions):
        pendulum_action = actions['pendulum']
        adv_action = 0.0
        if 'adversary{}'.format(self.curr_adversary) in actions.keys():
            adv_action = actions['adversary{}'.format(self.curr_adversary)]
            pendulum_action += adv_action * self.strengths[self.curr_adversary]
            pendulum_action = np.clip(pendulum_action, a_min=self.action_space.low, a_max=self.action_space.high)
        obs, reward, done, info = super().step(pendulum_action)

        if self.concat_actions:
            self.update_observed_obs(np.concatenate((obs, pendulum_action)))
        else:
            self.update_observed_obs(obs)

        info = {'pendulum': {'pendulum_reward': reward}}

        obs_dict = {'pendulum': self.observed_states}
        reward_dict = {'pendulum': reward}

        if self.kl_diff_training:
            for i in range(self.adversary_range):
                is_active = 1 if self.curr_adversary == i else 0
                obs_dict.update({
                    'adversary{}'.format(i): {'obs': self.observed_states, 'is_active': np.array([is_active])}
                })

                reward_dict.update({'adversary{}'.format(i): -reward})
        else:
            if self.adversary_range > 0:
                obs_dict.update({'adversary{}'.format(self.curr_adversary): self.observed_states})
                reward_dict.update({'adversary{}'.format(self.curr_adversary): -reward})

        done_dict = {'__all__': done}
        return obs_dict, reward_dict, done_dict, info

    def reset(self):
        obs = super().reset()
        if self.concat_actions:
            self.update_observed_obs(np.concatenate((obs, [0.0])))
        else:
            self.update_observed_obs(obs)

        curr_obs = {'pendulum': self.observed_states}
        if self.kl_diff_training:
            for i in range(self.adversary_range):
                is_active = 1 if self.curr_adversary == i else 0
                curr_obs.update({'adversary{}'.format(i):
                                     {'obs': self.observed_states,
                                      'is_active': np.array([is_active])
                                      }})
        else:
            if self.adversary_range > 0:
                curr_obs.update({'adversary{}'.format(self.curr_adversary): self.observed_states})

        return curr_obs


class ModelBasedPendulumEnv(PendulumEnv):
    def __init__(self, config):
        """Env with adversaries that are just sinusoids. Used for testing out identification schemes."""
        super(ModelBasedPendulumEnv, self).__init__(config)
        self.num_adversaries = config["num_adversaries"]
        self.adversary_strength = config["adversary_strength"]
        self.model_based = config["model_based"]
        self.guess_adv = config['guess_adv']
        # TODO(use an autoregressive model to condition this guess on your action)
        self.guess_next_state = config['guess_next_state']
        self.num_concat_states = config['num_concat_states']
        self.adversary_type = config['adversary_type']
        if self.adversary_type == 'state_func':
            self.state_weights = config['weights']
        elif self.adversary_type == 'rand_state_func':
            weight_size = super().observation_space.shape[0]
            self.state_weights = np.random.uniform(low=-1, high=1, size=weight_size)
        elif self.adversary_type == 'rand_friction':
            self.state_weights = np.array([0, 0, np.random.uniform(low=-1, high=0)])
        elif self.adversary_type == 'friction':
            self.state_weights = config['weights']
        # used to track the previously observed states
        self.observed_states = np.zeros(self.observation_space.shape[0])
        self.correct_adv_score = 5.0
        self.correct_state_coeff = 1.0
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)
        self.num_correct_guesses = 0
        self.state_error = np.zeros(super().observation_space.shape[0])
        # purely used for readout and visualization
        self.curr_state_error = np.zeros(super().observation_space.shape[0])
        # attribute that can be used to turn the adversary off for testing
        self.has_adversary = True
        # Attribute that can be used to overwrite memory for testing. If it's false the
        # state is just the currently observed state concatenated with zeros. If it's true
        # then we correctly append in prior past state
        self.use_memory = True
        # the reward without any auxiliary rewards
        self.true_rew = 0.0

    @property
    def observation_space(self):
        obs_space = super().observation_space
        action_space = super().action_space
        low = np.tile(np.concatenate((obs_space.low, action_space.low)), self.num_concat_states)
        high = np.tile(np.concatenate((obs_space.high, action_space.high)), self.num_concat_states)
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

        if self.has_adversary:
            if self.adversary_type == 'cos':
                adv_action = np.cos(2 * np.pi * self.curr_adversary * self.step_num * self.dt)
            elif self.adversary_type == 'state_func':
                adv_action = super()._get_obs() @ self.state_weights[self.curr_adversary]
            elif self.adversary_type == 'rand_state_func':
                adv_action = super()._get_obs() @ self.state_weights
            elif self.adversary_type == 'friction':
                adv_action = super()._get_obs() @ self.state_weights[self.curr_adversary]
            elif self.adversary_type == 'rand_friction':
                adv_action = super()._get_obs() @ self.state_weights
            else:
                sys.exit('The only supported adversary types are `state_func`, `cos`, `rand_state_func`,'
                         '`friction` and `rand_friction`')
            pendulum_action = adv_action * self.adversary_strength + torque
        else:
            pendulum_action = torque

        pendulum_action = np.clip(-self.max_torque, self.max_torque, pendulum_action)
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

        return self.update_observed_obs(np.concatenate((obs, torque))), rew, done, info

    def update_observed_obs(self, new_obs):
        """Add in the new observations and overwrite the stale ones"""
        original_shape = int(self.observation_space.shape[0] / self.num_concat_states)
        # If this is false we don't roll the states forward so the old states just get overwritten
        if self.use_memory:
            self.observed_states = np.roll(self.observed_states, shift=original_shape, axis=-1)
        self.observed_states[0: original_shape] = new_obs
        return self.observed_states

    def reset(self):
        self.num_correct_guesses = 0
        self.true_rew = 0.0

        self.curr_state_error = np.zeros(super().observation_space.shape[0])
        self.state_error = np.zeros(super().observation_space.shape[0])

        if self.adversary_type == 'rand_state_func':
            weight_size = super().observation_space.shape[0]
            self.state_weights = np.random.uniform(low=-1, high=1, size=weight_size)
        elif self.adversary_type == 'rand_friction':
            self.state_weights = np.array([0, 0, np.random.uniform(low=-1, high=0)])
        return self.update_observed_obs(np.concatenate((super().reset(), [0.0])))

    def select_new_adversary(self):
        self.curr_adversary = np.random.randint(low=0, high=self.num_adversaries)
