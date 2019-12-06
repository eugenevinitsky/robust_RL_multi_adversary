"""This code is adapted from https://github.com/vita-epfl/CrowdNav"""

import logging

import cv2
import gym
from gym.spaces import Box, Dict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import rvo2

from envs.utils.human import Human
from envs.utils.info import *
from envs.utils.utils import point_to_segment_dist
from utils.constants import ROBOT_COLOR, GOAL_COLOR, HUMAN_COLOR, BACKGROUND_COLOR, COLOR_LIST


class CrowdSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, robot):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.config = config

        # Training details
        self.num_stacked_frames = config.getint('train_details', 'num_stacked_frames')
        self.train_on_images = config.getboolean('train_details', 'train_on_images')
        self.show_images = config.getboolean('train_details', 'show_images')

        self.time_limit = config.getint('env', 'time_limit')
        self.discretization = config.getint('env', 'discretization')
        self.grid = np.linspace([-6, -6], [6, 6], self.discretization)
        self.robot_grid_size = np.maximum(int(0.1 / np.abs(self.grid[0, 0] - self.grid[1, 0])), 2)
        self.image = np.ones((self.discretization, self.discretization, 3)) * 255
        self.observed_image = np.ones((self.discretization, self.discretization, 3 * self.num_stacked_frames)) * 255
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.gauss_noise_state_stddev = config.getfloat('env', 'gaussian_noise_state_stddev')
        self.gauss_noise_action_stddev = config.getfloat('env', 'gaussian_noise_action_stddev')
        self.add_gauss_noise_state = config.getboolean('env', 'add_gaussian_noise_state')
        self.add_gauss_noise_action = config.getboolean('env', 'add_gaussian_noise_action')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.edge_discomfort_dist = config.getfloat('reward', 'edge_discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.edge_penalty = config.getfloat('reward', 'edge_penalty')
        self.closer_goal = config.getfloat('reward', 'closer_goal')
        self.randomize_goals = config.getboolean('sim', 'randomize_goals')
        self.update_goals = config.getboolean('sim', 'update_goals')

        # transfer configs
        self.change_colors_mode = config.get('transfer', 'change_colors_mode')
        self.chase_robot = config.getboolean('transfer', 'chase_robot')
        self.restrict_goal_region = config.getboolean('transfer', 'restrict_goal_region')

        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.accessible_space = config.getfloat('sim', 'accessible_space')
            self.goal_region = config.getfloat('sim', 'goal_region')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        self.robot = robot
        self.attention_weights = None
        self.obs_norm = 100
        self.v_lim = 0.2
        self.rad_lim = 1.2

        # generate a set of humans so we have something in the observation space
        self.generate_random_human_position(self.human_num, rule=self.train_val_sim)


    @property
    def observation_space(self):
        if not self.train_on_images:
            temp_obs = np.concatenate([human.get_observable_state().as_array() for human in self.humans])
            return Box(low=-1.0, high=1.0, shape=(temp_obs.shape[0] + 4, ))
        else:
            img_shape = self.image.shape
            new_tuple = (img_shape[0], img_shape[1], img_shape[2] * self.num_stacked_frames)
            return Box(low=-1.0, high=1.0, shape=new_tuple)


    @property
    def action_space(self):
        # TODO(@evinitsky) what are the right bounds
        return Box(low=-1.0 * self.time_step, high=1.0 * self.time_step, shape=(2, ))

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False                            
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).
        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        # Set up the colors
        self.robot_color = ROBOT_COLOR
        self.human_color = HUMAN_COLOR
        self.goal_color = GOAL_COLOR
        self.background_color = BACKGROUND_COLOR

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            if self.randomize_goals:
                random_goal = self.generate_random_goals()
                self.robot.set(0, 0, random_goal[0], random_goal[1], 0, 0, np.pi / 2)
            else:
                self.robot.set(0, 0, 0, self.circle_radius, 0, 0, np.pi / 2) #default goal is directly above robot


            # By setting np.random.seed with essentially the iteration number, they can ensure that the 
            # 'training cases' are the same each time.
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                ## this is how they handle 'testing', if you set test_case to -1 it'll set up a specific
                ## initial configuration
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = np.concatenate([human.get_observable_state().as_array() for human in self.humans]) / self.obs_norm
            normalized_pos = np.asarray(self.robot.get_position())/self.accessible_space
            normalized_goal = np.asarray(self.robot.get_goal_position())/self.accessible_space
            ob = np.concatenate((ob, list(normalized_pos), list(normalized_goal)))

        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        # TODO(@evinitsky) don't overwrite the ob, just calculate ob only once
        if self.train_on_images or self.show_images:
            bg_color = np.array(self.background_color)[np.newaxis, np.newaxis, :]
            self.image = np.ones((self.discretization, self.discretization, 3)) * bg_color

            # should we shift the colors?
            change_colors = False
            if self.change_colors_mode != 'no_change':
                change_colors = True
            self.image_state_space(update_colors=change_colors)

            # TODO(@evinitsky) don't recreate this every time
            img = np.rot90(self.image)
            # if needed, render the ob for visualization
        if self.show_images:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1000, 1000)
            cv2.imshow("image", img)
            cv2.waitKey(2)
        if self.train_on_images:
            # The last axis is represented as {s_{t}, ..., s_{t - num_stacked_frames + 2},
            #                                  s_{t - num_stacked_frames + 1}}
            # where each of these states is three channels wide.
            # We roll it forward so that we can overwrite the oldest frame
            self.observed_image = np.roll(self.observed_image, shift=3, axis=-1)
            self.observed_image[:, :, 0: 3] = img
            ob = (self.observed_image - 128.0) / 255.0

        if self.add_gauss_noise_state:
            ob = np.random.normal(scale=self.gauss_noise_state_stddev, size=ob.shape) + ob

        return np.clip(ob, a_min=self.observation_space.low, a_max=self.observation_space.high)

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.add_gauss_noise_action:
            action = action + (np.random.normal(scale=self.gauss_noise_action_stddev, size=action.shape) * self.time_step)
            action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            if self.chase_robot:
                human.set_goal([self.robot.px, self.robot.py]) #update goal of human to where robot is
            human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                robot_vx, robot_vy = action
                vx = human.vx - robot_vx
                vy = human.vy - robot_vy
            else:
                # rescale the actions so that they are within the bounds of the robot motions
                r, v = action
                r = r * self.rad_lim
                v = v * self.v_lim
                vx = human.vx - v * np.cos(r + self.robot.theta)
                vy = human.vy - v * np.sin(r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        cur_dist_to_goal = norm(self.robot.get_position() - np.array(self.robot.get_goal_position()))
        next_dist_to_goal = norm(end_position - np.array(self.robot.get_goal_position()))
        reaching_goal = next_dist_to_goal < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            if self.randomize_goals:
                new_goal = self.generate_random_goals()
                self.robot.set_goal(new_goal)
                #print("New Goal", self.robot.get_goal_position())
            if self.update_goals:
                done = False
            else:
                done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        #if too close to the edge, add penalty
        if (np.abs(np.abs(end_position) - self.accessible_space) < self.edge_discomfort_dist).any():
            reward += self.edge_penalty
        #if getting closer to goal, add reward
        if cur_dist_to_goal - next_dist_to_goal > 0.1:
            reward += self.closer_goal

        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = np.concatenate([human.get_observable_state().as_array() for human in self.humans])
                normalized_pos = np.asarray(self.robot.get_position()) / self.accessible_space
                normalized_goal = np.asarray(self.robot.get_goal_position()) / self.accessible_space
                ob = np.concatenate((ob, list(normalized_pos), list(normalized_goal))) / self.obs_norm
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = np.concatenate([human.get_next_observable_state(action).as_array()
                                 for human, action in zip(self.humans, human_actions)]) / self.obs_norm
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        # We need to create this ob if we are showing images OR training on images
        if self.show_images or self.train_on_images:
            bg_color = np.array(self.background_color)[np.newaxis, np.newaxis, :]
            self.image = np.ones((self.discretization, self.discretization, 3)) * bg_color

            # test if we should  shift the colors?
            change_colors = False
            if self.change_colors_mode == 'every_step':
                change_colors = True
            self.image_state_space(update_colors=change_colors)
            # TODO(@evinitsky) don't recreate this every time
            img = np.rot90(self.image)
            # if needed, render the ob for visualization
        if self.show_images:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1000, 1000)
            cv2.imshow("image", img)
            cv2.waitKey(2)
        if self.train_on_images:
            self.observed_image = np.roll(self.observed_image, shift=3, axis=-1)
            self.observed_image[:, :, 0: 3] = img
            ob = (self.observed_image - 128.0) / 255.0

        if self.add_gauss_noise_state:
            ob = np.random.normal(scale=self.gauss_noise_state_stddev, size=ob.shape) + ob

        return np.clip(ob, a_min=self.observation_space.low, a_max=self.observation_space.high), reward, done, {}


    def image_state_space(self, update_colors):
        """Take the current state and render it as an image
        Parameters
        ==========
        update_colors: bool
            If true, we change the color scheme

        Returns
        =======
        None
        """
        if update_colors:
            self.robot_color = COLOR_LIST[np.random.randint(len(COLOR_LIST))]
            self.human_color = COLOR_LIST[np.random.randint(len(COLOR_LIST))]
            self.goal_color = COLOR_LIST[np.random.randint(len(COLOR_LIST))]
            self.background_color = COLOR_LIST[np.random.randint(len(COLOR_LIST))]

        # Fill in the robot position
        robot_pos = self.robot.get_full_state().position
        self.fill_grid(self.pos_to_coord(robot_pos), self.robot_grid_size, self.robot_color)

        # Fill in the human position
        for human in self.humans:
            pos = human.get_full_state().position
            self.fill_grid(self.pos_to_coord(pos), self.robot_grid_size, self.human_color)

        # Fill in the goal image
        goal_pos = self.robot.get_goal_position()
        self.fill_grid(self.pos_to_coord(goal_pos), self.robot_grid_size, self.goal_color)

    def pos_to_coord(self, pos):
        """Convert a position into a coordinate in the image

        Parameters
        ==========
        pos: np.ndarray w/ shape (2,)

        Returns
        =======
        coordinates: list of length 2
        """
        # TODO (@evinitsky) remove magic numbers
        # find the coordinates of the closest grid point
        min_idx = np.argmin(np.abs(self.grid - pos), axis=0)
        return min_idx

    def fill_grid(self, coordinate, radius, color):
        """Fill in the grid with a square of length 2 * radius

        Parameters
        ==========
        coordinate: np.ndarray w/ shape (2,)
            the center coordinate in the grid
        radius: int
            half of the length of the square we fill in around the pos
        color: np.ndarray (3,)
            The color to fill in

        Returns
        =======
        None

        """
        x_left, x_right = np.maximum(coordinate[0] - radius, 0), np.minimum(coordinate[0] + radius, self.grid.shape[0])
        y_left, y_right = np.maximum(coordinate[1] - radius, 0), np.minimum(coordinate[1] + radius, self.grid.shape[0])
        self.image[x_left: x_right, y_left: y_right, :] = color

    def render(self, mode='human', output_file=None):

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.accessible_space, self.accessible_space)
            ax.set_ylim(-self.accessible_space, self.accessible_space)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal_positions = [state[0].goal_position for state in self.states]

            goal = plt.Circle(goal_positions[0], radius=self.robot.radius, color=goal_color, fill=True, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                goal.center = goal_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                #TODO: Fix this visualization code
                plt.show(block=False)
                plt.pause(3)
                plt.close()
        else:
            raise NotImplementedError
            
    def generate_random_goals(self):
        if self.restrict_goal_region:
            goal_reg = self.goal_region
        else:
            goal_reg = self.accessible_space #unrestricted goal region, goal can be anywhere in accessible space
        return (np.random.rand(2) - 0.5) * 2 * goal_reg

class MultiAgentCrowdSimEnv(CrowdSimEnv, MultiAgentEnv):

    def __init__(self, config, robot):
        super(MultiAgentCrowdSimEnv, self).__init__(config, robot)
        self.adversary_action_scaling = config.getfloat('env', 'adversary_action_scaling')
        self.adversary_state_scaling = config.getfloat('env', 'adversary_state_scaling')
        self.perturb_actions = config.getboolean('ma_train_details', 'perturb_actions')
        self.perturb_state = config.getboolean('ma_train_details', 'perturb_state')
        self.num_iters = 0
        # We don't want to perturb until we actually have a reasonably good policy to start with
        self.adversary_start_iter = int(4e4)
        self.num_adversaries = 0
        # self.curr_adversary = 0
        if not self.perturb_state and not self.perturb_actions:
            logging.exception("Either one of perturb actions or perturb state must be true")

    # @property
    # def adv_observation_space(self):
    #     """
    #     Simple action space for an adversary that can perturb
    #     every element of the agent's observation space.
    #
    #     Therefore, its action space is the same size as the agent's
    #     observation space.
    #     """
    #     obs_size = super().observation_space.shape
    #     dict_space = Dict({'obs': Box(low=-1.0, high=1.0, shape=obs_size, dtype=np.float32),
    #                        'is_active': Box(low=-1.0, high=1.0, shape=(1,), dtype=np.int32)})
    #     return dict_space

    @property
    def adv_observation_space(self):
        """
        Simple action space for an adversary that can perturb
        every element of the agent's observation space.

        Therefore, its action space is the same size as the agent's
        observation space.
        """
        obs_size = super().observation_space.shape
        box_space =  Box(low=-1.0, high=1.0, shape=obs_size, dtype=np.float32)
        return box_space

    @property
    def adv_action_space(self):
        """
        Simple action space for an adversary that can perturb
        every element of the agent's observation space.

        Therefore, its action space is the same size as the agent's
        observation space.
        """
        obs_size = super().observation_space.shape
        if len(obs_size) > 1:
            obs_size = np.product(obs_size)
        else:
            obs_size = obs_size[0]
        act_size = super().action_space.shape[0]
        shape = obs_size * self.perturb_state + act_size * self.perturb_actions
        box = Box(low=-1.0 * self.time_step, high=1.0 * self.time_step, shape=(shape,), dtype=np.float32)
        return box

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        # adversary_key = 'adversary{}'.format(self.curr_adversary)
        adversary_key = 'adversary'

        robot_action = action['robot']
        if self.num_iters > self.adversary_start_iter:
            if self.perturb_state and self.perturb_actions:
                action_perturbation = action[adversary_key][:2] * self.adversary_action_scaling
                state_perturbation = action[adversary_key][2:] * self.adversary_state_scaling
            elif not self.perturb_state and self.perturb_actions:
                action_perturbation = action[adversary_key] * self.adversary_action_scaling
            else:
                state_perturbation = action[adversary_key] * self.adversary_state_scaling

            if self.perturb_actions:
                robot_action = action['robot'] + action_perturbation
                # apply clipping so that it can't exceed the bounds of what the robot can do
                robot_action = np.clip(robot_action, a_min=self.action_space.low, a_max=self.action_space.high)

        ob, reward, done, info = super().step(robot_action, update)

        curr_obs = {'robot': np.clip(ob,
                               a_min=self.observation_space.low[0],
                               a_max=self.observation_space.high[0])}
        reward_dict = {'robot': reward}

        if self.num_iters > self.adversary_start_iter:
            # for i in range(self.num_adversaries):
        #             #     is_active = 1 if self.curr_adversary == i else 0
        #             #     curr_obs.update({'adversary{}'.format(i): {'obs': ob, 'is_active': np.array([is_active])}})
        #             #
        #             # if self.perturb_state:
        #             #     curr_obs['robot'] = np.clip(curr_obs['robot'] + state_perturbation,
        #             #                                 a_min=self.observation_space.low,
        #             #                                 a_max=self.observation_space.high)
        #             #
        #             # reward_dict.update({'adversary{}'.format(i): -reward for i in range(self.num_adversaries)})
            curr_obs.update({'adversary': ob})

            if self.perturb_state:
                curr_obs['robot'] = np.clip(curr_obs['robot'] + state_perturbation,
                                            a_min=self.observation_space.low,
                                            a_max=self.observation_space.high)

            reward_dict.update({'adversary': -reward})

        done = {'__all__': done}
        
        return curr_obs, reward_dict, done, info

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.num_iters += 1
        # self.curr_adversary = int(np.random.randint(low=0, high=self.num_adversaries))
        ob = super().reset(phase, test_case)
        if self.num_iters > self.adversary_start_iter:
            curr_obs = {'robot': ob, 'adversary': ob}
        else:
            curr_obs = {'robot': ob}
        #     for i in range(self.num_adversaries):
        #         is_active = 1 if self.curr_adversary == i else 0
        #         curr_obs.update({'adversary{}'.format(i):
        #                              {'obs': np.clip(ob, a_min=self.observation_space.low, a_max=self.observation_space.high),
        #                               'is_active': np.array([is_active])}})
        return curr_obs
        