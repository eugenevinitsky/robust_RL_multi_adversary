import numpy as np
from numpy.linalg import norm
import abc
import logging
from envs.policy.policy_factory import policy_factory
from envs.utils.action import ActionXY, ActionRot
from envs.utils.state import ObservableState, FullState
from geometry_msgs.msg import Twist

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
        self.px = 0.0
        self.py = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.time_step = 0.0
        self.reward = 0
        self.accessible_space = config.getfloat('sim', 'accessible_space')

        # If true all of the actions will be slightly less than intended by a constant factor
        self.friction = config.getboolean('transfer', 'friction')
        self.friction_coef = config.getfloat('transfer', 'friction_coef')

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

    def set_goal(self, goal):
        """
        Set goal of agent
        :param goal: Tuple of (gx,gy) for goal x and y positions
        :return: None
        """
        self.gx = goal[0]
        self.gy = goal[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        pass
        # assert action.shape[0] == 2, "Actions don't have shape 2"

    def check_valid_action(self, new_px, new_py, cur_px, cur_py, world_dim):
        """
        Check if action will take the agent out of the grid space
        :param new_px: new x position
        :param new_py: new y position
        :param world_dim: size of the world in one dimension
        :return: Valid px, py
        """
        if np.abs(new_px) > world_dim:
            new_px = cur_px
        if np.abs(new_py) > world_dim:
            new_py = cur_py

        return new_px, new_py

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            vx, vy = action
            if self.friction:
                vx = vx - (self.friction_coef * vx)
                vy = vy - (self.friction_coef * vy)
            px = self.px + vx * delta_t
            py = self.py + vy * delta_t
        else:
            r, v = action
            if self.friction:
                r = r - (np.pi / 2) * self.friction_coef * r
                v = v - self.friction_coef * v
            theta = self.theta + r
            px = self.px + np.cos(theta) * v * delta_t
            py = self.py + np.sin(theta) * v * delta_t

        return self.check_valid_action(px,py, self.px, self.py, self.accessible_space)

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            vx, vy = action
            self.vx = vx
            self.vy = vy
        else:
            r, v = action
            self.theta = (self.theta + r) % (2 * np.pi)
            self.vx = v * np.cos(self.theta)
            self.vy = v * np.sin(self.theta)

    def get_cmd_vel_(self):
        vel_msg = Twist()
        vel_msg.linear.x = self.vx
        vel_msg.linear.y = self.vy
        return vel_msg

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

    def get_reward(self):
        return self.reward

    def set_reward(self, reward):
        self.reward = reward


