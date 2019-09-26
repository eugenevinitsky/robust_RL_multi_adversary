import numpy as np
from numpy.linalg import norm
import abc
import logging
from envs.policy.policy_factory import policy_factory
from envs.utils.action import ActionXY, ActionRot
from envs.utils.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = self.policy.kinematics if self.policy is not None else None

        self.friction = config.get('transfer', 'friction')
        self.friction_coef = config.getfloat('transfer', 'friction_coef')

        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        pass
        # assert action.shape[0] == 2, "Actions don't have shape 2"

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            ax, ay = action
        else:
            theta_dot, v_accel = action
            # TODO(@evinitsky) what's the right way to do this, presumably the rotation doesn't happen first
            self.theta = (self.theta + theta_dot * delta_t) % (2 * np.pi)
            ax = v_accel * np.cos(self.theta)
            ay = v_accel * np.sin(self.theta)

        if self.friction:
            self.vx = self.vx + ax * delta_t - 2 * self.friction_coef * self.vx * delta_t
            self.vy = self.vy + ay * delta_t - 2 * self.friction_coef * self.vy * delta_t
        else:
            self.vx = self.vx + ax * delta_t
            self.vy = self.vy + ay * delta_t

        self.px = self.px + self.vx * delta_t + 0.5 * ax * (delta_t ** 2)
        self.py = self.py + self.vy * delta_t + 0.5 * ay * (delta_t ** 2)


    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        self.compute_position(action, self.time_step)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

