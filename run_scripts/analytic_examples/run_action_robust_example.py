"""This runs a multi-adversary formulation of the example from
`Robust Reinforcement Learning viaAdversarial training with Langevin Dynamics`"""

import argparse
import sys

import numpy as np

import matplotlib.pyplot as plt


# problem is ((1 - \alpha) a + \alpha b)^2
# note, agent is a, adversary is b
def agent_grad(b, a):
    return 2 * (1 - alpha) * ((1 - alpha) * a + alpha * b)


def adv_grad(b, a):
    return -(2 * alpha * ((1 - alpha) * a + alpha * b))


def f(b, a):
    return ((1 - alpha) * a  + alpha * b) ** 2


clip = 1.0
std_dev = 0.2
alpha = 0.2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_adversaries', type=int, default=1,
                        help='Number of adversaries to use')
    parser.add_argument('--num_gd_step', type=int, default=50,
                        help='Number of gradient steps')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate of gradient descent')
    parser.add_argument('--vector_plot', action='store_true', default=False,
                        help='If true, plot the vector plot on the figure')
    parser.add_argument('--contour', action='store_true', default=False,
                        help='If true, plot the contour plot on the figure')
    parser.add_argument('--ne_distance', type=str, default='close',
                        help='Options are [close, far]. This controls how close the initial values are '
                             'to Nash.')
    parser.add_argument('--eg', action='store_true', default=False,
                        help='If true, use the extra gradient method from the paper')
    parser.add_argument('--plot_all_adv', action='store_true', default=False,
                        help='If true, plot all the omegas')
    parser.add_argument('--std_dev', action='store_true', default=False,
                        help='If true, add gaussian noise')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of times to run this and make plots')

    args = parser.parse_args()
    ne_dist = args.ne_distance
    num_advs = args.num_adversaries

    # construct the initial values
    if ne_dist == 'close':
        low = -0.25
        high = 0.25
    elif ne_dist == 'far':
        low = -1.0
        high = 1.0
    elif ne_dist == 'fixed_far':
        low = 0.9
        high = 0.9
    else:
        sys.exit('ne_dist must be one of `close` or `far`')

    for q in range(args.num_trials):
        np.random.seed(q * 10)

        agent_init = np.random.uniform(low=low, high=high)
        adv_init = []
        for num_adv in range(num_advs):
            adv_init.append(np.random.uniform(low=low, high=high))

        agent_vals = np.zeros(args.num_gd_step + 1)
        agent_vals[0] = np.random.uniform(low=low, high=high)
        print(agent_vals[0])
        adv_vals = np.zeros((args.num_gd_step + 1, num_advs))
        for num_adv in range(num_advs):
            adv_vals[0, num_adv] = np.random.uniform(low=low, high=high)

        print(agent_vals[0], adv_vals[0, 0])

        for step_num in range(1, args.num_gd_step + 1):
            agent_grad_arr = np.zeros(num_advs)
            adv_grad_arr = np.zeros(num_advs)
            for adv_num in range(num_advs):
                agent_grad_arr[adv_num] = agent_grad(adv_vals[step_num - 1, adv_num], agent_vals[step_num - 1])
                adv_grad_arr[adv_num] = adv_grad(adv_vals[step_num - 1, adv_num], agent_vals[step_num - 1])

            if args.eg:
                theta_temp = np.clip(args.lr * np.mean(agent_grad_arr) + agent_vals[step_num - 1], -clip, clip)
                omega_temp = np.clip(args.lr * adv_grad_arr + adv_vals[step_num - 1, :], -clip, clip)
                temp_agent_grad_arr = np.zeros(num_advs)
                temp_adv_grad_arr = np.zeros(num_advs)
                for adv_num in range(num_advs):
                    temp_agent_grad_arr[adv_num] = agent_grad(omega_temp[adv_num], theta_temp)
                    temp_adv_grad_arr[adv_num] = adv_grad(omega_temp[adv_num], theta_temp)
                agent_vals[step_num] = np.clip(args.lr * np.mean(temp_agent_grad_arr) + agent_vals[step_num - 1], -clip, clip)
                adv_vals[step_num, :] = np.clip(args.lr * temp_adv_grad_arr + adv_vals[step_num - 1, :], -clip, clip)
                if args.std_dev:
                    agent_vals[step_num] += args.lr * np.random.normal(loc=0.0, scale=std_dev)
                    adv_vals[step_num] += args.lr * np.random.normal(loc=0.0, scale=std_dev)
            else:
                agent_vals[step_num] = np.clip(args.lr * np.mean(agent_grad_arr) + agent_vals[step_num - 1], -clip, clip)
                adv_vals[step_num, :] = np.clip(args.lr * adv_grad_arr + adv_vals[step_num - 1, :], -clip, clip)
                if args.std_dev:
                    agent_vals[step_num] += args.lr * np.random.normal(loc=0.0, scale=std_dev)
                    adv_vals[step_num] += args.lr * np.random.normal(loc=0.0, scale=std_dev)

        plt.figure()
        if args.plot_all_adv:
            for adv_num in range(num_advs):
                plt.scatter(adv_vals[0, adv_num], agent_vals[0], s=100, c='r')
                plt.scatter(adv_vals[:, adv_num], agent_vals, s=5)
                plt.scatter(adv_vals[-1, adv_num], agent_vals[-1], s=100, c='y')
        else:
            plt.scatter(adv_vals[0, 0], agent_vals[0], s=100, c='r')
            plt.scatter(adv_vals[:, 0], agent_vals)
            plt.scatter(adv_vals[-1, 0], agent_vals[-1], s=100, c='y')

        if args.vector_plot:
            x, y = np.meshgrid(np.linspace(-1.0, 1.0, 10), np.linspace(-1.0, 1.0, 10))
            u = adv_grad(x, y)
            v = agent_grad(x, y)
            plt.quiver(x, y, u, v)

        if args.contour:
            x, y = np.meshgrid(np.linspace(-1.0, 1.0, 10), np.linspace(-1.0, 1.0, 10))
            z = f(x, y)
            plt.pcolor(x, y, z, alpha=0.5)

        plt.xlabel('b (adversary)')
        plt.ylabel('a (agent)')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        print(0.5 * f(-1, 1) + 0.5 * f(1, 1))
        print(f(-1, 1))

        plt.savefig('figures/action_robust_example_{}'.format(q))
