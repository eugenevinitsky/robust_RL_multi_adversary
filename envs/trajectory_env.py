"""This code is adapted from https://github.com/vita-epfl/CrowdNav"""

import logging
import sys

import cv2
import gym
from gym.spaces import Box, Dict, Tuple, Discrete

if sys.platform == 'darwin':
    try:
        import matplotlib

        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib import patches
    except:
        pass
import numpy as np
from numpy.linalg import norm
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import rvo2

from envs.utils.human import Human
from envs.utils.info import *
from envs.utils.utils import point_to_segment_dist
from utils.constants import ROBOT_COLOR, GOAL_COLOR, HUMAN_COLOR, BACKGROUND_COLOR, COLOR_LIST


class TrajectoryEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, robot1, robot2, robot3, robot4):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.config = config

        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')

        # HARDCODED CONSTANTS. MOVE THESE OUT
        self.robot1 = robot1
        self.robot2 = robot2
        self.robot3 = robot3
        self.robot4 = robot4

        self.closer_goal = config.getfloat('reward', 'closer_goal')

        self.robots = [self.robot1, self.robot2, self.robot3, self.robot4]

        self.attention_weights = None
        self.obs_norm = 100
        self.v_lim = 0.2
        self.rad_lim = 1.2
        self.iter_num = 0

        self.accessible_space_x = config.getfloat('sim', 'accessible_space_x')
        self.accessible_space_y = config.getfloat('sim', 'accessible_space_y')

        self.discretization = config.getint('env', 'discretization')
        self.grid = np.linspace([-self.accessible_space_x, -self.accessible_space_y],
                                [self.accessible_space_x, self.accessible_space_y], self.discretization)
        self.robot_grid_size = np.maximum(int(self.robot1.radius / np.abs(self.grid[0, 0] - self.grid[1, 0])), 2)

    @property
    def observation_space(self):
            num_obs = 10  # goal pos, robot pos
            # TODO(@evinitsky) enable
            # num_obs = 5 # goal pos, robot pos, heading

            return Box(low=-1.0, high=1.0, shape=(num_obs,))

    @property
    def action_space(self):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        # increment the counter
        self.iter_num += 1
        self.states = {}
        self.dones = {}

        # Set up the colors
        self.robot_color = ROBOT_COLOR
        self.goal_color = GOAL_COLOR
        self.background_color = BACKGROUND_COLOR

        if self.robot1 is None or self.robot2 is None or self.robot3 is None or self.robot4 is None:
            raise AttributeError('robots have to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        rand_heading = np.random.random() * 2 * np.pi

        goal_x = [-self.accessible_space_x, self.accessible_space_x, self.accessible_space_x, -self.accessible_space_x]
        goal_y = [-self.accessible_space_y, self.accessible_space_y, -self.accessible_space_y, self.accessible_space_y]

        for robot in self.robots:
            robot.set(0, 0, goal_x, goal_y, 0, 0, rand_heading)
            robot.time_step = self.time_step

        obs = {}

        # get current observation
        for robot in self.robots:
            self.dones[robot.id] = False
            self.states[robot] = [robot.get_full_state()]
            if robot.sensor == 'coordinates':
                normalized_pos = np.asarray(robot.get_position()) / np.asarray(
                    [self.accessible_space_x, self.accessible_space_y])
                normalized_goal = np.asarray(robot.get_goal_position()).T / np.asarray(
                    [self.accessible_space_x, self.accessible_space_y])
                normalized_goal = normalized_goal.flatten()

                ob = np.concatenate((normalized_pos, normalized_goal))
                # theta = np.asarray([self.robot.theta]) / (2 * np.pi)
                # ob = np.concatenate((ob, list(normalized_pos), list(normalized_goal), theta))
                np.clip(ob, a_min=self.observation_space.low, a_max=self.observation_space.high)
                obs[robot.id] = ob

            elif robot.sensor == 'RGB':
                raise NotImplementedError

        return obs

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def transform_actions(self, action):
        """Returns the action transformed into appropriate units and coordinates"""
        r, v = np.copy(action)
        # scale r to be between - self.rad_lim and self.rad_lim
        r = r * self.rad_lim
        v = v * self.v_lim
        scaled_action = np.array([r, v])
        return scaled_action

    def step(self, action_dict, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        obs = {}
        rews = {}
        done = True

        for robot in self.robots:
            action = action_dict[robot.id]
            # rescale the actions so that they are within the bounds of the robot motions
            scaled_action = self.transform_actions(action)
            r, v = scaled_action

            # check if reaching the goal
            end_position = np.array(robot.compute_position(scaled_action, self.time_step))

            goal_x, goal_y = robot.get_goal_position()
            reached_goal = False
            for goal in np.concatenate((goal_x, goal_y)):
                cur_dist_to_goal = norm(robot.get_position() - goal)
                next_dist_to_goal = norm(end_position - goal)
                reached_goal = next_dist_to_goal < robot.radius

            if self.global_time >= self.time_limit - 1:
                rews[robot.id] = 0
                self.dones[robot.id] = True
            elif reached_goal:
                rews[robot.id] = self.success_reward
                self.dones[robot.id] = True
            else:
                rews[robot.id] = 0
                self.dones[robot.id] = False
                done = False

            rews[robot.id] += self.closer_goal * (cur_dist_to_goal - next_dist_to_goal)

            if update:
                # store state, action value and attention weights
                self.states[robot].append([robot.get_full_state()])
                # update all agents
                if self.dones[robot.id]: #if robot is at goal, continue
                    obs[robot.id] = np.zeros((10,))
                    continue

                robot.step(scaled_action)

                self.global_time += self.time_step

                # compute the observation
                if robot.sensor == 'coordinates':
                    normalized_pos = np.asarray(robot.get_position()) / np.asarray(
                        [self.accessible_space_x, self.accessible_space_y])
                    normalized_goal = np.asarray(robot.get_goal_position()).T / np.asarray(
                        [self.accessible_space_x, self.accessible_space_y])
                    normalized_goal = normalized_goal.flatten()

                    ob = np.concatenate((normalized_pos, normalized_goal))
                    # theta = np.asarray([self.robot.theta]) / (2 * np.pi)
                    # ob = np.concatenate((ob, list(normalized_pos), list(normalized_goal), theta))
                    np.clip(ob, a_min=self.observation_space.low, a_max=self.observation_space.high)
                    obs[robot.id] = ob
                elif robot.sensor == 'RGB':
                    raise NotImplementedError
        all_done = {}
        all_done['__all__'] = done

        return obs, rews, all_done, {}

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
            self.fill_grid(self.pos_to_coord(pos), self.human_grid_size, self.human_color)

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
        pred_robot_color = 'blue'
        pred_arrow_color = 'green'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        fig, ax = plt.subplots(figsize=(20 * self.accessible_space_x, 20 * self.accessible_space_y))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-self.accessible_space_x, self.accessible_space_x)
        ax.set_ylim(-self.accessible_space_y, self.accessible_space_y)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        # add robots and its goal

        all_robot_pos = {}
        all_robot_artists = {}
        goal_artists = []


        for robot in self.robots:
            robot_states = self.states[robot]
            all_robot_pos[robot] = [state.position for state in robot_states]

            robot_plt = plt.Circle(all_robot_pos[robot][0], self.robot.radius, fill=True, color=robot_color)
            all_robot_artists[robot] = robot_plt
            ax.add_artist(robot_plt)

        goals = [(self.accessible_space_x, self.accessible_space_y),
                     (-self.accessible_space_x, -self.accessible_space_y),
                     (self.accessible_space_x, -self.accessible_space_y),
                     (-self.accessible_space_x, self.accessible_space_y)]

        for goal in goals:
            goal_plt = plt.Circle(goal, radius=self.robot1.radius, color=goal_color, fill=True,
                                  label='Goal')
            goal_artists.append(goal_plt)

        plt.legend([robot for robot in all_robot_artists.values()] + goals, ['Robot1', 'Robot2', 'Robot3', 'Robot4', 'Goal'], fontsize=16)

        # add time annotation
        time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
        ax.add_artist(time)

        global_step = 0

        def update(frame_num):
            nonlocal global_step
            global_step = frame_num

            for robot in self.robots:
                all_robot_artists[robot].center = all_robot_pos[robot][frame_num]


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

