"""adaptive.py

An object-oriented approach to implementing
different adaptive strategies.

"""

import collections
import numpy as np
from visualize.linear_env.robust_adaptive_lqr.python import utils
import time

from abc import abstractmethod
from visualize.linear_env.robust_adaptive_lqr.python.nominal import NominalStrategy
from visualize.pendulum.run_rollout import instantiate_rollout, DefaultMapping
from utils.rllib_utils import get_config_from_path

checkpoint_path = "/Users/eugenevinitsky/Desktop/Research/Data/sim2real/linear/" \
                  "04-13-2020/linear_dr_d3_h200_r4/linear_dr_d3_h200_r4/PPO_0_lambda=0.5," \
                  "lr=0.0005_2020-04-14_06-16-3346eyb1_c/"
checkpoint_num = "250"


class RLStrategy(NominalStrategy):
    """The base class for all adaptive methods

    The way to use this class is as follows:

    e = MyAdaptiveMethod(...)
    e.reset(rng)
    e.prime(num_iters, static_feedback, rng)
    for _ in range(horizon):
        cur_regret = e.step(rng)
        # do something with current regret
        # ...

    """

    def __init__(self,
                 Q,
                 R,
                 A_star,
                 B_star,
                 sigma_w,
                 rls_lam,
                 sigma_explore,
                 reg,
                 epoch_multiplier,
                 epoch_schedule='linear'):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam,
                         sigma_explore,
                         reg,
                         epoch_multiplier,
                         'linear')

        # initialize the env and the agent
        rllib_config, checkpoint = get_config_from_path(checkpoint_path, checkpoint_num)
        rllib_config['num_envs_per_worker'] = 1
        self.env, self.agent, self.multiagent, self.use_lstm, self.policy_agent_mapping, \
        self.state_init, self.action_init = \
            instantiate_rollout(rllib_config, checkpoint)
        self.mapping_cache = {}  # in case policy_agent_mapping is stochastic
        self.agent_id = 'agent'
        self.prev_actions = DefaultMapping(
            lambda agent_id: self.action_init[self.mapping_cache[agent_id]])
        self.agent_states = DefaultMapping(
            lambda agent_id: self.state_init[self.mapping_cache[agent_id]])
        self.prev_action = np.zeros(self._A_star.shape[0])
        self.prev_rewards = collections.defaultdict(lambda: 0.)


    def _get_input(self, state, rng):
        """Obtain the next input to play from the current state"""
        return self.get_action(state)

    def get_action(self, state):
        obs = np.concatenate((self.Ahat.flatten(), self.Bhat.flatten(), self.fixed_K.flatten(), state,
                              self.prev_action))
        policy_id = self.mapping_cache.setdefault(
            self.agent_id, self.policy_agent_mapping(self.agent_id))
        p_use_lstm = self.use_lstm[policy_id]
        if not p_use_lstm:
            a_action = self.agent.compute_action(
                obs,
                prev_action=self.prev_actions[self.agent_id],
                prev_reward=self.prev_rewards[self.agent_id],
                policy_id=policy_id)
        else:
            a_action, p_state, _ = self.agent.compute_action(
                obs,
                state=self.agent_states[self.agent_id],
                prev_action=self.prev_actions[self.agent_id],
                prev_reward=self.prev_rewards[self.agent_id],
                policy_id=policy_id)
            self.agent_states[self.agent_id] = p_state
        self.prev_actions[self.agent_id] = a_action
        self.prev_action = a_action
        return a_action

    def reset(self, rng):
        """Reset both the dynamics and internal state.

        Must be called before ether prime() or step() is called.
        """
        self._state_history = []
        self._input_history = []
        self._transition_history = []
        self._cost_history = []

        # tracks the estimate errors for each epoch
        self._error_history = []

        # tracks the length of epochs
        self._epoch_history = []

        # tracks the average infinite time horizon cost
        self._infinite_horizon_cost_history = []

        if self._rls is not None:
            logger = self._get_logger()
            logger.debug("Using RLS estimator with rls_lam={}".format(self._rls_lam))
            self._rls = utils.RecursiveLeastSquaresEstimator(self._n, self._p, self._rls_lam)
        self._rls_history = []

        self._regret = 0
        self._epoch_idx = 0
        self._iteration_idx = 0
        self._iteration_within_epoch_idx = 0
        self._state_cur = np.zeros((self._n,))
        self._last_reset_time = time.time()
        self._has_primed = False
        self._explore_stddev_history = []

        # reset the things you need to reset
        self.mapping_cache = {}  # in case policy_agent_mapping is stochastic
        self.agent_id = 'agent'
        self.prev_actions = DefaultMapping(
            lambda agent_id: self.action_init[self.mapping_cache[agent_id]])
        self.agent_states = DefaultMapping(
            lambda agent_id: self.state_init[self.mapping_cache[agent_id]])
        self.prev_action = np.zeros(self._A_star.shape[0])
        self.prev_rewards = collections.defaultdict(lambda: 0.)

    def prime(self, num_iterations, static_feedback, excitation, rng):
        """Initialize the adaptive method with rollouts

        Must be called after reset() and before step() is called

        """

        super().prime(num_iterations, static_feedback, excitation, rng)

        # prime the network with an action
        self.fixed_K = self._current_K
        self.get_action(np.zeros_like(self._state_cur))
