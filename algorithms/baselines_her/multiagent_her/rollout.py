from collections import deque

import numpy as np
import pickle

from algorithms.baselines_her.multiagent_her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.selected_adversary = 0
        self.Q_history = {}
        for key in policy.keys():
            self.Q_history.update({key:deque(maxlen=history_len) })

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], {key: [] for key in self.policy.keys()}, [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = {key: [] for key in self.policy.keys()}
        u_dict = {}

        if len(self.policy) > 1:
            temp_policy_dict = {'agent': self.policy['agent'],
                                'adversary{}'.format(self.selected_adversary):
                                    self.policy['adversary{}'.format(self.selected_adversary)]}
        else:
            temp_policy_dict = self.policy

        for t in range(self.T):
            for key, policy in temp_policy_dict.items():
                policy_output = policy.get_actions(
                    o, ag, self.g,
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)

                if self.compute_Q:
                    u, Q = policy_output
                    Qs[key].append(Q)
                else:
                    u = policy_output

                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
                u_dict[key] = u

            # compute new states and observations
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)

            # TODO(@evinitsky) remove this and handle non-multiagent case better
            try:
                obs_dict_new, _, done, info = self.venv.step(u_dict)
            except:
                obs_dict_new, _, done, info = self.venv.step(u_dict['agent'][0])

            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([info['is_success']])

            if (isinstance(done, dict) and done['__all__']) or (not isinstance(done, dict) and done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break

            # TODO(@evinitsky) put back, make useful
            # for i, info_dict in enumerate(info):
            #     for idx, key in enumerate(self.info_keys):
            #         try:
            #             # TODO(@evinitsky) hack, fix
            #             if key == 'agent':
            #                 info_values[idx][t, i] = info[i][key]['agent_reward']
            #             else:
            #                 info_values[idx][t, i] = info[i][key]
            #         except:
            #             import ipdb; ipdb.set_trace()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            if isinstance(done, dict):
                dones.append(np.array([done['__all__']]))
            else:
                dones.append(done)

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            for key, val in u_dict.items():
                acts[key].append(u_dict[key].copy())
            goals.append(self.g.copy()[np.newaxis, :])
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode_dict = {}
        for key in temp_policy_dict.keys():
            episode_dict.update({key: dict(o=obs,
                           u=acts[key],
                           g=goals,
                           ag=achieved_goals)})
        for agent_id in temp_policy_dict.keys():
            for key, value in zip(self.info_keys, info_values):
                dict_key = 'info_{}'.format(key)
                if 'adversary' in agent_id:
                    dict_key = dict_key.replace('agent', agent_id)
                episode_dict[agent_id][dict_key] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            for key in temp_policy_dict.keys():
                self.Q_history[key].append(np.mean(Qs[key]))
        self.n_episodes += self.rollout_batch_size

        # TODO(@evinitsky) do we need this?
        for key, episode in episode_dict.items():
            episode_dict[key] = convert_episode_to_batch_major(episode)

        return episode_dict

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        for key in self.policy.keys():
            self.Q_history[key].clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            for key in self.policy.keys():
                logs += [('mean_Q_{}'.format(key), np.mean(self.Q_history[key]))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

