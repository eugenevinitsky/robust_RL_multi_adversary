# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/eugenevinitsky/Desktop/Research/Code/adversarial_sim2real/algorithms/baselines_her/multiagent_her/dummy_vec_env.py
# Compiled at: 2020-05-17 17:37:14
# Size of source mod 2**32: 2986 bytes
import numpy as np
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

class DummyVecEnv(VecEnv):
    __doc__ = '\n    VecEnv that does runs multiple environments sequentially, that is,\n    the step and reset commands are send to one environment at a time.\n    Useful when debugging and when num_env == 1 (in the latter case,\n    avoids communication overhead)\n    '

    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = {k:np.zeros(((self.num_envs,) + tuple(shapes[k])), dtype=(dtypes[k])) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=(np.bool))
        self.buf_rews = np.zeros((self.num_envs,), dtype=(np.float32))
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = ''

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, 'actions {} is either not a list or has a wrong size - cannot match to {} environments'.format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)

        return (
         self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
         self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)

        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        import ipdb
        ipdb.set_trace()
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)
# okay decompiling dummy_vec_env.cpython-36.pyc
