# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/eugenevinitsky/Desktop/Research/Code/adversarial_sim2real/algorithms/baselines_her/multiagent_her/wrappers.py
# Compiled at: 2020-05-17 16:50:42
# Size of source mod 2**32: 1128 bytes
import gym, numpy as np

class TimeLimit(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return (
         observation, reward, done, info)

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return (self.env.reset)(**kwargs)


class ClipActionsWrapper(gym.Wrapper):

    def step(self, action):
        action = np.nan_to_num(action)
        if isinstance(action, dict):
            for key, val in action.items():
                action[key] = np.clip(val, self.action_space.low, self.action_space.high)

        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return (self.env.reset)(**kwargs)
# okay decompiling wrappers.cpython-36.pyc
