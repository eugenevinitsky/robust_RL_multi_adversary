"""Agent wants to navigate to a goal, adversary selects the starting point"""

import sys

# import cv2
import gym
from gym.spaces import Box, Dict
import numpy as np
from ray.rllib.env import MultiAgentEnv

if sys.platform == 'darwin':
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib import patches
        from matplotlib.collections import PatchCollection
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
    except:
        pass

class GoalEnv(MultiAgentEnv, gym.Env):

    def __init__(self, config):
        self.config = config
        self.range = 6
        self.x_lim = np.array([-self.range, self.range])
        self.y_lim = np.array([-self.range, self.range])
        self.goal_pos = np.array([0, 2 * self.range / 3])

        self.show_image = False

        # This sets how many adversaries exist at each strength level
        self.num_adv_strengths = config["num_adv_strengths"]
        # This sets how many adversaries exist per strength level
        self.advs_per_strength = config["advs_per_strength"]
        # whether the adversaries are receiving penalties for being too similar
        self.l2_reward = config['l2_reward']
        self.l2_reward_coeff = config['l2_reward_coeff']

        # This sets whether we should use adversaries across a reward range
        self.reward_range = config["reward_range"]
        # This sets the adversaries low reward range
        self.low_reward = config["low_reward"]
        # This sets wthe adversaries high reward range
        self.high_reward = config["high_reward"]
        self.reward_targets = np.linspace(start=self.low_reward, stop=self.high_reward,
                                          num=self.num_adv_strengths)
        # repeat the bins so that we can index the adversaries easily
        self.reward_targets = np.repeat(self.reward_targets, self.advs_per_strength)

        # agent position
        self.curr_pos = np.array([0, 0])
        self.speed = 1.0
        self.radius = 0.3

        self.horizon = config["horizon"]
        self.step_num = 0

        self.adversary_range = self.num_adv_strengths * self.advs_per_strength
        self.curr_adversary = 0
        self.total_rew = 0

    @property
    def observation_space(self):
        return Box(low=-self.range, high=self.range, shape=(2, ))

    @property
    def action_space(self):
        return Box(low=-self.speed, high=self.speed, shape=(2, ))

    @property
    def adv_observation_space(self):
        if self.l2_reward:
            dict_space = Dict({'obs': self.observation_space,
                               'is_active': Box(low=-1.0, high=1.0, shape=(1,), dtype=np.int32)})
            return dict_space
        else:
            return Box(low=-self.range, high=self.range, shape=(2,))

    @property
    def adv_action_space(self):
        return Box(low=-self.range, high=self.range, shape=(2, ))


    def step(self, action_dict):
        if self.step_num == 0:
            self.curr_pos = action_dict['adversary{}'.format(self.curr_adversary)]
        else:
            self.curr_pos += action_dict['agent']

        self.curr_pos = np.clip(self.curr_pos, self.observation_space.low, self.observation_space.high)

        done = np.linalg.norm(self.curr_pos - self.goal_pos) < self.radius
        if self.step_num > self.horizon:
            done = True

        curr_obs = {'agent': self.curr_pos}
        base_rew = -1
        self.total_rew += base_rew
        curr_rew = {'agent': base_rew}

        if self.reward_range:
            adv_reward = [-1 * np.abs((float(self.step_num) / self.horizon) * self.reward_targets[
                i] - self.total_reward) for i in range(self.adversary_range)]
        else:
            adv_reward = [-self.total_rew for _ in range(self.adversary_range)]

        # the adversaries get observations on the final steps and on the first step
        if self.step_num == 0 or done:
            if self.l2_reward:
                is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                curr_obs.update({
                    'adversary{}'.format(i): {"obs": self.curr_pos, "is_active": np.array([is_active[i]])}
                    for i in range(self.adversary_range)})
                curr_rew.update({
                    'adversary{}'.format(i): adv_reward[i]
                    for i in range(self.adversary_range)})
            else:
                curr_obs.update({
                    'adversary{}'.format(self.curr_adversary): self.curr_pos
                })
                curr_rew.update({
                    'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]
                })

        info = {'agent': {'agent_reward': base_rew}}

        done_dict = {'__all__': done}

        if self.show_image:
            self.render()

        self.step_num += 1
        return curr_obs, curr_rew, done_dict, info


    def reset(self):
        self.step_num = 0
        self.total_rew = 0
        self.curr_pos = np.array([0, 0])

        curr_obs = {'agent': self.curr_pos}
        if self.l2_reward:
            is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
            curr_obs.update({
                'adversary{}'.format(i): {"obs": self.curr_pos, "is_active": np.array([is_active[i]])}
                for i in range(self.adversary_range)})
        else:
            curr_obs.update({
                'adversary{}'.format(self.curr_adversary): self.curr_pos
            })

        return curr_obs

    def select_new_adversary(self):
        if self.adversary_range > 0:
            # the -1 corresponds to not having any adversary on at all
            self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)

    def render(self, mode='human'):
        fig = plt.figure(figsize=(8, 8))
        # fig = Figure()
        # canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_xlim([-self.range, self.range])
        ax.set_ylim([-self.range, self.range])

        # draw the agent
        circle1 = plt.Circle((self.curr_pos[0], self.curr_pos[1]), self.radius, color='r')

        # draw the goal position
        circle2 = plt.Circle((self.goal_pos[0], self.goal_pos[1]), self.radius, color='blue')

        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # plt.show()
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        img = np.array(fig.canvas.renderer.buffer_rgba())


        # canvas.draw()
        # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1000, 1000)
        # cv2.imshow("image", img)
        # cv2.waitKey(1)
