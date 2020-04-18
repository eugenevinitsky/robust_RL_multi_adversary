"""This file creates a baseline using certainty equivalent control to compare against as a baseline"""

import numpy as np
import time
import logging
import sys
from multiprocessing import Pool, cpu_count

import ray
from scipy.stats import ortho_group

from visualize.linear_env.robust_adaptive_lqr.python import utils
from visualize.linear_env.robust_adaptive_lqr.python import examples
from visualize.linear_env.robust_adaptive_lqr.python.optimal import OptimalStrategy
from visualize.linear_env.robust_adaptive_lqr.python.nominal import NominalStrategy
from visualize.linear_env.robust_adaptive_lqr.python.ofu import OFUStrategy
from visualize.linear_env.robust_adaptive_lqr.python.sls import SLS_FIRStrategy, SLS_CommonLyapunovStrategy, sls_common_lyapunov, SLSInfeasibleException
from visualize.linear_env.robust_adaptive_lqr.python.ts import TSStrategy
from visualize.linear_env.robust_adaptive_lqr.python.rl_agent import RLStrategy

import matplotlib.pylab as plt
import seaborn as sns
sns.set_style('ticks')

trials_per_method = 1000
prime_horizon = 100
prime_excitation = 1.0
horizon = 200
dim = 6
eigv_mag = 0.4
A = -0.8 * np.eye(dim)
B = np.eye(dim)
Q = 10 * np.eye(dim)
R = np.eye(dim)
sigma_w = 1.0
sigma_excitation = 0.1

logging.basicConfig(level=logging.WARN)

prime_seed = 45727

def optimal_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return OptimalStrategy(Q=Q, R=R, A_star=A_star, B_star=B_star, sigma_w=sigma_w)

def nominal_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return NominalStrategy(Q=Q,
                          R=R,
                          A_star=A_star,
                          B_star=B_star,
                          sigma_w=sigma_w,
                          sigma_explore=sigma_excitation,
                          reg=1e-5,
                          epoch_multiplier=10, rls_lam=None)

def rl_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return RLStrategy(Q=Q,
                      R=R,
                      A_star=A_star,
                      B_star=B_star,
                      sigma_w=sigma_w,
                      sigma_explore=sigma_excitation,
                      reg=1e-5,
                      epoch_multiplier=10, rls_lam=None)

def ofu_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return OFUStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  reg=1e-5,
                  actual_error_multiplier=1, rls_lam=None)

def ts_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return TSStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  reg=1e-5,
                  tau=500,
                  actual_error_multiplier=1, rls_lam=None)

def sls_fir_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return SLS_FIRStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  sigma_explore=sigma_excitation,
                  reg=1e-5,
                  epoch_multiplier=10,
                  truncation_length=12,
                  actual_error_multiplier=1, rls_lam=None)

def sls_cl_ctor(Q, R, A_star, B_star, sigma_w, sigma_excitation):
    return SLS_CommonLyapunovStrategy(Q=Q,
                  R=R,
                  A_star=A_star,
                  B_star=B_star,
                  sigma_w=sigma_w,
                  sigma_explore=sigma_excitation,
                  reg=1e-5,
                  epoch_multiplier=10,
                  actual_error_multiplier=1, rls_lam=None)

def run_one_trial(new_env_ctor, seed, prime_fixed=False):
    rng = np.random.RandomState(seed)
    if prime_fixed: # reducing variance
        rng_prime = np.random.RandomState(prime_seed)
    else:
        rng_prime = rng
    eigs = np.random.uniform(low=-eigv_mag, high=eigv_mag, size=dim)
    diag_mat = np.diag(eigs)
    # now sample some unitary matrices
    orthonormal_mat = ortho_group.rvs(dim)
    perturbation_matrix = orthonormal_mat.T @ diag_mat @ orthonormal_mat
    env = new_env_ctor(Q, R, A + perturbation_matrix, B, sigma_w, sigma_excitation)
    env.reset(rng_prime)
    _, K_init = utils.dlqr(A, B, 1e-3*np.eye(dim), np.eye(dim))
    env.prime(prime_horizon, K_init, prime_excitation, rng_prime)
    regret = np.array([env.step(rng) for _ in range(horizon)])
    env.complete_epoch(rng)
    err, cost = env.get_statistics(iteration_based=True)
    return regret, err, cost

# def run_one_trial_with_actor(env, seed, prime_fixed=False):
#     rng = np.random.RandomState(seed)
#     if prime_fixed: # reducing variance
#         rng_prime = np.random.RandomState(prime_seed)
#     else:
#         rng_prime = rng
#     env.reset(rng_prime)
#     env.prime(prime_horizon, K_init, prime_excitation, rng_prime)
#     regret = np.array([env.step(rng) for _ in range(horizon)])
#     env.complete_epoch(rng)
#     err, cost = env.get_statistics(iteration_based=True)
#     return regret, err, cost

def spawn_invocation(method, p, prime_fixed=False):
    seed = np.random.randint(0xFFFFFFFF)
    ctor = {
        'optimal': optimal_ctor,
        'nominal': nominal_ctor,
        'ofu': ofu_ctor,
        'ts': ts_ctor,
        'sls_fir': sls_fir_ctor,
        'sls_cl': sls_cl_ctor,
        "rl": rl_ctor
    }[method]
    return (p.apply_async(run_one_trial, (ctor, seed, prime_fixed)), seed)

def process_future_list(ftchs):
    regrets = []
    errors = []
    costs = []
    seeds = []
    bad_invocations = 0
    for ftch, seed in ftchs:
        try:
            reg, err, cost = ftch.get()
        except Exception as e:
            bad_invocations += 1
            continue
        regrets.append(reg)
        errors.append(err)
        costs.append(cost)
        seeds.append(seed)
    return np.array(regrets), np.array(errors), np.array(costs), np.array(seeds), bad_invocations

def get_errorbars(regrets, q=10, percent_bad=0):
    median = np.percentile(regrets, q=50-percent_bad, axis=0)
    low10 = np.percentile(regrets, q=q, axis=0)
    high90 = np.percentile(regrets, q=100-(q-percent_bad), axis=0)
    return median, low10, high90

def plot_list_medquantile(datalist, legendlist=None, xlabel=None, ylabel=None, semilogy=False,
                          loc='upper left', alpha=0.1, figsize=(8,4)):
    rgblist = sns.color_palette('viridis', len(datalist))
    plt.figure(figsize=figsize)
    for idx, data in enumerate(datalist):
        median, lower, higher = data
        if semilogy:
            plt.semilogy(range(len(median)), median, color=rgblist[idx], label=legendlist[idx])
        else:
            plt.plot(range(len(median)), median, color=rgblist[idx], label=legendlist[idx])
        plt.fill_between(np.array(np.arange(len(median))), median.astype(np.float),
                        higher.astype(np.float), color=rgblist[idx], alpha=alpha)
    if legendlist is not None:
        plt.legend(loc=loc)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()



if __name__=='__main__':
    # strategies = ['optimal', 'nominal', 'ofu', 'ts']  # , 'sls_cl', 'sls_fir']
    strategies = ['nominal']  # , 'sls_cl', 'sls_fir']
    start_time = time.time()
    with Pool(processes=cpu_count()) as p:
        all_futures = [[spawn_invocation(method, p, prime_fixed=True)
                        for _ in range(trials_per_method)] for method in strategies]
        list_of_results = [process_future_list(ftchs) for ftchs in all_futures]
    print("finished execution in {} seconds".format(time.time() - start_time))

    regretlist = []
    costs_list = []

    # strat_rearranged =  [strategies[2], strategies[3], strategies[1], strategies[0]]
    # res_rearranged =  [list_of_results[2], list_of_results[3], list_of_results[1], list_of_results[0]]

    # strat_rearranged = ['rl', strategies[2], strategies[3], strategies[1], strategies[0]]
    # res_rearranged = [list_of_results[4], list_of_results[2], list_of_results[3], list_of_results[1],
    #                   list_of_results[0]]
    strat_rearranged = [strategies[0]]
    res_rearranged = [list_of_results[0]]

    for name, result in zip(strat_rearranged, res_rearranged):
        regrets, errors, costs, _, bad_invocations = result
        print(name, "bad_invocations", bad_invocations)
        percent_bad = bad_invocations / trials_per_method * 100
        regretlist.append(get_errorbars(regrets, q=10, percent_bad=percent_bad))
        costs_list.append(get_errorbars(costs, q=10, percent_bad=percent_bad))

    sns.set_palette("muted")
    plot_list_medquantile(regretlist, legendlist=strat_rearranged, xlabel="Iteration", ylabel="Regret")
    plot_list_medquantile(costs_list[:-1], legendlist=strat_rearranged[:-1], xlabel="Iteration",
                          ylabel="Cost Suboptimality", semilogy=True, loc='upper right')
    # run_one_trial(optimal_ctor, 0, prime_fixed=False)