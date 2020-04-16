"""Agent wants to navigate to a goal, adversary selects the starting point"""

import sys

try:
    import cv2
except:
    pass
import gym
from gym.spaces import Box, Dict
import numpy as np
from ray.rllib.env import MultiAgentEnv
import scipy
from scipy.linalg import solve_discrete_are
from scipy.stats import ortho_group

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

class LinearEnv(MultiAgentEnv, gym.Env):

    def __init__(self, config):
        self.config = config
        self.dim = config['dim']
        self.scaling = config['scaling']
        self.agent_strength = np.abs(config['agent_strength'])
        # this is how much cost we put on actions. Backwards compatibility, remove the get later.
        self.action_cost_coeff = config['action_cost_coeff']
        # we create an env that's stable but close to the RHP
        self.A = np.identity(self.dim) * (- np.abs(self.scaling))
        self.B = np.identity(self.dim)
        self.Q = 10 * np.identity(self.dim)
        self.R = np.identity(self.dim) * self.action_cost_coeff
        # Noise in actions
        self.sigma_w = 1.0
        self.prime_excitation = 1.0
        self.perturbation_matrix = np.zeros(self.dim)
        self.exit_penalty = 50000
        self.scale_factor = 50

        # agent starting position
        self.start_pos = np.zeros(self.dim)
        self.curr_pos = self.start_pos

        self.horizon = config["horizon"]
        self.rollout_length = config["rollout_length"]

        self.step_num = 0
        self.total_rew = 0

        # this is just for visualizing the position
        self.show_image = False
        self.radius = 0.2
        self.image_array = []

        self.num_concat_states = config["num_concat_states"]

        self.adversary_strength = config["adversary_strength"]
        # This sets how many adversaries exist at each strength level
        self.num_adv_strengths = config["num_adv_strengths"]
        # This sets how many adversaries exist per strength level
        self.advs_per_strength = config["advs_per_strength"]
        # whether the adversaries are receiving penalties for being too similar
        self.l2_reward = config['l2_reward']
        self.l2_reward_coeff = config['l2_reward_coeff']
        self.l2_in_tranche = config['l2_in_tranche']
        self.l2_memory = config['l2_memory']
        self.l2_memory_target_coeff = config['l2_memory_target_coeff']
        # if true we sample eigenvalues uniiformly instead of matrix entries
        self.eigval_rand = config['eigval_rand']

        self.adversary_range = self.num_adv_strengths * self.advs_per_strength
        self.curr_adversary = 0

        # This sets whether we should use adversaries across a reward range
        self.reward_range = config["reward_range"]
        # This sets the adversaries low reward range
        self.low_reward = config["low_reward"]
        # This sets wthe adversaries high reward range
        self.high_reward = config["high_reward"]
        self.num_adv_rews = config['num_adv_rews']
        self.advs_per_rew = config['advs_per_rew']
        self.reward_targets = np.linspace(start=self.low_reward, stop=self.high_reward,
                                          num=self.num_adv_rews)
        # repeat the bins so that we can index the adversaries easily
        self.reward_targets = np.repeat(self.reward_targets, self.advs_per_rew)
        # this tells us if we should compute regret as the reward or LQR cost as the reward
        self.regret_reward = config['regret']
        self.optimal_K = np.zeros((self.dim, self.dim))
        # this tells us if we should reset the rollout or just continue. If true,
        # our actions are also added as perturbations to a stabilizing controller
        self.should_reset = config['should_reset']

        print('reward targets are', self.reward_targets)

        # keep track of the actions that they did take
        self.action_list = []

        # construct the adversaries that we actually compare against for l2 distance.
        # we only compare with adversaries that are in the same self.advs_per_rew tranche.
        self.comp_adversaries = []
        for i in range(self.adversary_range):
            curr_tranche = int(i / self.advs_per_rew)
            low_range = max(curr_tranche * self.advs_per_rew, i - self.advs_per_rew)
            high_range = min((curr_tranche + 1) * self.advs_per_rew, i + self.advs_per_rew)
            self.comp_adversaries.append([low_range, high_range])

        # instantiate the l2 memory tracker
        if self.adversary_range > 0 and self.l2_memory:
            self.global_l2_memory_array = np.zeros((self.adversary_range, self.adv_action_space.low.shape[0]))
            self.local_l2_memory_array = np.zeros((self.adversary_range, self.adv_action_space.low.shape[0]))
            self.local_num_observed_l2_samples = np.zeros(self.adversary_range)

        # track past states
        self.observed_states = np.zeros(self.observation_space.low.shape[0])

        # flag we can use to turn perturbation off
        self.should_perturb = True

    def update_observed_obs(self, new_obs):
        """Add in the new observations and overwrite the stale ones"""
        original_shape = new_obs.shape[0]
        self.observed_states = np.roll(self.observed_states, shift=original_shape, axis=-1)
        self.observed_states[0: original_shape] = new_obs
        return self.observed_states

    def update_global_action_mean(self, mean_array):
        self.global_l2_memory_array = (1 - self.l2_memory_target_coeff) * self.global_l2_memory_array + self.l2_memory_target_coeff * mean_array
        self.local_l2_memory_array = np.zeros((self.adversary_range, self.adv_action_space.low.shape[0]))
        self.local_num_observed_l2_samples = np.zeros(self.adversary_range)

    def get_observed_samples(self):
        return self.local_l2_memory_array, self.local_num_observed_l2_samples

    @property
    def observation_space(self):
        if self.should_reset:
            return Box(low=-np.inf, high=np.inf, shape=((self.dim + self.action_space.low.shape[0]) * self.num_concat_states, ))
        else:
            # here we see the K matrix and then the state, action pair at each time-step, eigenvalues of A
            return Box(low=-np.inf, high=np.inf, shape=(3 * self.dim**2 + self.action_space.low.shape[0] + self.dim + self.dim, ))

    @property
    def action_space(self):
        return Box(low=-self.agent_strength, high=self.agent_strength, shape=(self.dim, ))

    @property
    def adv_observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.dim,))

    @property
    def adv_action_space(self):
        return Box(low=-self.adversary_strength, high=self.adversary_strength, shape=(int(self.dim ** 2), ))

    def step(self, action_dict):

        # the rollout gets reset periodically to prevent the reward from going off to infinity
        if self.step_num % self.rollout_length == 0 and self.should_reset:
            self.curr_pos = self.start_pos

        if self.step_num == 0 and self.adversary_range > 0:
            self.perturbation_matrix = action_dict['adversary{}'.format(self.curr_adversary)].reshape((self.dim, self.dim))
            # store this since the adversary won't get a reward until the last step
            if self.l2_reward and not self.l2_memory:
                self.action_list = [action_dict['adversary{}'.format(i)] for i in range(self.adversary_range)]
            if self.l2_memory and self.l2_reward:
                self.action_list = [action_dict['adversary{}'.format(self.curr_adversary)]]
                self.local_l2_memory_array[self.curr_adversary] += action_dict['adversary{}'.format(self.curr_adversary)]
        elif self.step_num == 0 and self.adversary_range == 0 and self.should_perturb:
            if self.eigval_rand:
                eigs = np.random.uniform(low=-self.adversary_strength, high=self.adversary_strength, size=self.dim)
                diag_mat = np.diag(eigs)
                # now sample some unitary matrices
                orthonormal_mat = ortho_group.rvs(self.dim)
                self.perturbation_matrix = orthonormal_mat.T @ diag_mat @ orthonormal_mat
            else:
                self.perturbation_matrix = np.random.uniform(low=-self.adversary_strength, high=self.adversary_strength,
                                                             size=self.adv_action_space.low.shape[0]).reshape((self.dim, self.dim))
        elif self.step_num == 0 and not self.should_perturb:
            self.perturbation_matrix = np.zeros((self.dim, self.dim))

        if self.step_num == 0 and self.regret_reward:
            # NOTE, this does not have sign included. It's not really important though since the
            # sign doesn't show up in the regret
            self.X = solve_discrete_are(self.A + self.perturbation_matrix, self.B, self.Q, self.R)
            self.Pstar, self.optimal_K = self.dlqr(self.A, self.B, self.Q, self.R)
            _, self.rollout_controller = self.dlqr(self.A, self.B, 1e-3*np.eye(self.dim), np.eye(self.dim))
            assert self.spectral_radius(self.A + self.B.dot(self.rollout_controller)) < 1
            # _J_star is the optimal infinite time reward
            self._J_star = (self.sigma_w ** 2) * np.trace(self.Pstar)

            # # Note that these do not necessarily yield stable eigenvalues if the action cost is very high
            # self.optimal_K = self.estimate_K(self.horizon + 1, self.A + self.perturbation_matrix, self.B)
            # for i in range(self.horizon + 1):
            # if np.any(np.linalg.eigvals(self.A + self.perturbation_matrix - self.B @ self.optimal_K) < -1 ):
            #     import ipdb; ipdb.set_trace()
            #     sys.exit("Something is messed up with your discrete ARE, it is returning unstable systems")

        # Okay, now if we are doing the pure regret formulation without resets, we do a rollout of length 100
        # from a stabilizing controller to give us initial estimates of A and B
        if not self.should_reset and self.step_num == 0 and self.regret_reward:
            curr_pos = self.start_pos
            x_mat = []
            u_mat = []
            trans_mat = []
            for i in range(100):
                x_mat.append(curr_pos)
                inp = self.rollout_controller @ curr_pos + self.prime_excitation * np.random.normal(size=(self.dim,))
                u_mat.append(inp)
                xt_1 = (self.A + self.perturbation_matrix) @ curr_pos + self.B @ inp + \
                    self.sigma_w * np.random.normal(size=(self.dim,))
                trans_mat.append(xt_1)
                curr_pos = xt_1
            # Now synthesize the A and B estimates
            self.Ahat, self.Bhat = self._design_controller(
                np.array(x_mat),
                np.array(u_mat),
                np.array(trans_mat))
            self.eig_A = np.abs(np.linalg.eigvals(self.Ahat))

        # if np.any(np.linalg.eigvals(self.A) < 1.1):
        #     print('curr pos', self.curr_pos)
        #     print('action', action_dict)
        #     print('current reward', self.total_rew)

        # dynamics update
        if self.should_reset:
            self.curr_pos = (self.A + self.perturbation_matrix) @ self.curr_pos + self.B @ action_dict['agent'] + \
                            self.sigma_w * np.random.normal(size=(self.dim,))
        else:
            self.curr_pos = (self.A + self.perturbation_matrix + np.diag(action_dict['agent'])) @ self.curr_pos + \
                            self.sigma_w * np.random.normal(size=(self.dim,))
        self.update_observed_obs(np.concatenate((self.curr_pos, action_dict['agent'])))

        if self.should_reset:
            curr_obs = {'agent': self.observed_states}
        else:
            curr_obs = {'agent': np.concatenate((self.Ahat.flatten(), self.Bhat.flatten(), self._current_K.flatten(),
                                                 self.eig_A.flatten(),
                                                 self.curr_pos, action_dict['agent']))}
        if self.regret_reward:
            # we treat the action as a diagonal K matrix
            agent_cost = self.curr_pos.T @ self.Q  @ self.curr_pos + action_dict['agent'].T @ self.R @ action_dict['agent']
            # this is the negative regret since we are going to maximize this quantity. We wouldn't want our
            # agent to maximize regret!
            regret = self._J_star - agent_cost
            base_rew = regret / self.scale_factor
        else:
            # LQR cost with Q and R being the identity. We don't take the square to keep the costs in reasonable size
            base_rew = -(np.linalg.norm(self.curr_pos)) - self.action_cost_coeff * (np.linalg.norm(action_dict['agent']))
        # print(self.total_rew)

        done = False
        # if we go off to infinity the episode should end probably
        if self.step_num == self.horizon or np.linalg.norm(self.curr_pos) > 20:
            done = True

        # penalize exiting
        if np.linalg.norm(self.curr_pos) > 20:
            curr_rew = {'agent': -self.exit_penalty / self.scale_factor}
            self.total_rew += -self.exit_penalty / self.scale_factor
        else:
            curr_rew = {'agent': base_rew}
            self.total_rew += base_rew

        if self.adversary_range > 0 and self.curr_adversary >= 0:

            # the adversaries get observations on the final steps and on the first step
            if done:
                if self.reward_range:
                    # we don't want to give the adversaries an incentive to end the rollout early
                    # so we make a positively shaped reward that peaks at self.reward_targets[i]
                    adv_reward = [-np.abs(self.reward_targets[i] - self.total_rew) for i in range(self.adversary_range)]
                else:
                    adv_reward = [-self.total_rew for _ in range(self.adversary_range)]

                if self.l2_reward:
                    # the adversary only takes actions at the first step so
                    # update the reward dict
                    # row index is the adversary, column index is the adversaries you're diffing against
                    # Also, we only want to do the l2 reward against the adversary in a given reward tranch

                    # If we are using the l2_memory, we diff against the mean vector instead of an action vector
                    if not self.l2_memory:
                        if self.l2_in_tranche:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in
                                  self.action_list[self.comp_adversaries[i][0]: self.comp_adversaries[i][1]]]
                                 for i, action_i in enumerate(self.action_list)])
                        else:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in self.action_list]
                                 for action_i in self.action_list])
                        l2_dists_mean = np.sum(l2_dists, axis=-1)
                    else:
                        if self.l2_in_tranche:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in
                                  self.global_l2_memory_array[self.comp_adversaries[i][0]: self.comp_adversaries[i][1]]]
                                 for i, action_i in enumerate(self.action_list)])
                        else:
                            l2_dists = np.array(
                                [[np.linalg.norm(action_i - action_j) for action_j in self.global_l2_memory_array]
                                 for action_i in self.action_list])
                        l2_dists_mean = np.sum(l2_dists)

                    if self.l2_memory:
                        curr_obs.update({
                            'adversary{}'.format(self.curr_adversary): self.curr_pos})
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]
                                        + l2_dists_mean * self.l2_reward_coeff}
                        curr_rew.update(adv_rew_dict)
                    else:
                        is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                        curr_obs.update({
                            'adversary{}'.format(i): {"obs": self.curr_pos, "is_active": np.array([is_active[i]])}
                            for i in range(self.adversary_range)})
                        # we get rewarded for being far away for other agents
                        adv_rew_dict = {'adversary{}'.format(i): adv_reward[i] + l2_dists_mean[i] *
                                                                 self.l2_reward_coeff for i in range(self.adversary_range)}
                        curr_rew.update(adv_rew_dict)
                else:
                    curr_obs.update({
                        'adversary{}'.format(self.curr_adversary): self.curr_pos})
                    adv_rew_dict = {'adversary{}'.format(self.curr_adversary): adv_reward[self.curr_adversary]}
                    curr_rew.update(adv_rew_dict)

        info = {'agent': {'agent_reward': base_rew}}

        done_dict = {'__all__': done}

        if self.show_image:
            self.render()

        if done and self.show_image:
            width, height, layers = self.image_array[0].shape
            out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, (width, height))

            for i in range(len(self.image_array)):
                out.write(self.image_array[i][:, :, 0:3])
            out.release()

        self.step_num += 1
        return curr_obs, curr_rew, done_dict, info

    def reset(self):
        self.step_num = 0
        self.total_rew = 0
        self.curr_pos = self.start_pos

        if self.should_reset:
            self.update_observed_obs(np.concatenate((self.curr_pos, [0.0] * self.dim)))
            curr_obs = {'agent': self.observed_states}
        else:
            curr_obs = {'agent': np.zeros(self.observation_space.shape[0])}

        if self.adversary_range > 0:
            if self.l2_reward and not self.l2_memory:
                is_active = [1 if i == self.curr_adversary else 0 for i in range(self.adversary_range)]
                curr_obs.update({
                    'adversary{}'.format(i): {"obs": self.curr_pos, "is_active": np.array([is_active[i]])}
                    for i in range(self.adversary_range)})
            else:
                curr_obs.update({
                    'adversary{}'.format(self.curr_adversary): self.curr_pos
                })
                if self.l2_memory:
                    self.local_num_observed_l2_samples[self.curr_adversary] += 1

        self.image_array = []

        if self.show_image:
            self.render()

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
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])

        # draw the agent
        circle1 = plt.Circle((self.curr_pos[0], self.curr_pos[1]), self.radius, color='r')

        # draw the goal position
        circle2 = plt.Circle((0, 0), self.radius, color='blue')

        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # plt.show()
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        img = np.array(fig.canvas.renderer.buffer_rgba())
        self.image_array.append(img)
        plt.close(fig)

        # # canvas.draw()
        # # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        #
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1000, 1000)
        # cv2.imshow("image", img)
        # cv2.waitKey(2)

    def estimate_K(self, horizon, A, B):
        """Solve for K recursively. Note that this returns -K so you don't need to add a sign to the matrix product with the state."""
        Q, R = self.Q, self.R
        # Calculate P matrices first for each step
        P_matrices = np.zeros((horizon + 1, Q.shape[0], Q.shape[1]))
        P_matrices[horizon] = Q
        for i in range(horizon - 1, 0, -1):
            P_t = P_matrices[i + 1]
            P_matrices[i] = Q + (A.T @ P_t @ A) - (A.T @ P_t @ B @ np.matmul(np.linalg.inv(R + B.T @ P_t @ B), B.T @ P_t @ A))
        # Hardcoded shape of K, change to inferred shape for diverse testing
        K_matrices = np.zeros((horizon, self.dim, self.dim))
        for i in range(horizon):
            P_i = P_matrices[i + 1]
            K_matrices[i] = -np.matmul(np.linalg.inv(R + B.T @ P_i @ B), B.T @ P_i @ A)
        return K_matrices

    def _design_controller(self, states, inputs, transitions):


        # do a least squares fit and controller based on the nominal
        Anom, Bnom, _ = self.solve_least_squares(states, inputs, transitions, reg=1e-5)
        _, K = self.dlqr(Anom, Bnom, self.Q, self.R)
        self._current_K = K

        # for debugging purposes,
        # check to see if this controller will stabilize the true system
        rho_true = self.spectral_radius(self.A + self.B.dot(self._current_K))
        # print("_design_controller: rho(A_* + B_* K)={}".format(
        #     rho_true))

        return (Anom, Bnom)

    def solve_least_squares(self, states, inputs, transitions, reg=0.0):
        """Solve for system dynamics from states and inputs
        """

        assert len(states.shape) == 2
        assert len(inputs.shape) == 2
        assert len(transitions.shape) == 2
        assert states.shape[0] == inputs.shape[0]
        assert states.shape == transitions.shape

        n, p = states.shape[1], inputs.shape[1]

        X = np.hstack((states, inputs))
        Y = transitions

        regI = reg * np.eye(n + p)

        Cov = X.T.dot(X) + regI

        # (n+p) x n
        Theta_hat = scipy.linalg.solve(Cov, X.T.dot(Y), sym_pos=False)
        # n x (n+p)
        Theta_hat = Theta_hat.T

        A_est = Theta_hat[:, :n]
        B_est = Theta_hat[:, n:]

        return A_est, B_est, Cov

    def dlqr(self, A, B, Q=None, R=None):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # ref Bertsekas, p.151
        if Q is None:
            Q = np.eye(A.shape[0])
        if R is None:
            R = np.eye(B.shape[1])

        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = -scipy.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A), sym_pos=True)

        A_c = A + B.dot(K)
        TOL = 1e-5
        if self.spectral_radius(A_c) >= 1 + TOL:
            print("WARNING: spectral radius of closed loop is:", self.spectral_radius(A_c))

        return P, K

    def LQR_cost(self, A, B, K, Q, R, sigma_w):
        """Compute infinite time horizon average LQR cost.
        Returns +inf if A+BK is not stable
        """

        L = A + B.dot(K)
        if self.spectral_radius(L) >= 1:
            return np.inf

        M = Q + K.T.dot(R).dot(K)

        P = self.solve_discrete_lyapunov(L, M)

        return (sigma_w ** 2) * np.trace(P)

    def spectral_radius(self, A):
        assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
        return max(np.abs(np.linalg.eigvals(A)))

    def solve_discrete_lyapunov(self, A, Q, method=None):
        """Solve A^T P A - P + Q = 0
        """

        # newer versions of scipy solve A P A^T - P + Q = 0,
        # while older ones solve A^T P A - P + Q = 0. I do not
        # remember exactly which version of scipy made the change.

        # I am going to assume you have the newer version installed.
        # If the assertion below fails, please add an if statement
        # that branches on your version.

        P = scipy.linalg.solve_discrete_lyapunov(A.T, Q, method)

        assert np.allclose(A.T.dot(P).dot(A) - P, -Q)

        return P