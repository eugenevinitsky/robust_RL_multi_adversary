# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/eugenevinitsky/Desktop/Research/Code/adversarial_sim2real/algorithms/baselines_her/multiagent_her/transfer_tests.py
# Compiled at: 2020-05-26 20:35:46
# Size of source mod 2**32: 4681 bytes
import os, matplotlib.pyplot as plt, numpy as np
num_samples = 3
slide_mass_sweep = np.linspace(0.5, 3.5, num_samples)
slide_friction_sweep = np.linspace(0.01, 0.2, num_samples)
push_friction_sweep = np.linspace(0.1, 2.0, num_samples)
push_mass_sweep = np.linspace(0.5, 3.5, num_samples)

def set_mass(env, mass):
    bnames = env.sim.model.body_names
    bindex = bnames.index('object0')
    env.sim.model.body_mass[bindex] = mass


def set_friction(env, fric):
    bnames = env.sim.model.geom_names
    bindex = bnames.index('object0')
    env.sim.model.geom_friction[bindex] = [fric * env.sim.model.geom_friction[bindex][0],
     fric * env.sim.model.geom_friction[bindex][1],
     fric * env.sim.model.geom_friction[bindex][2]]


def run_rollout(env, model):
    rew_total = 0
    obs = env.reset()
    done = False
    info = {}
    while not done:
        actions, _, _, _ = model.step(obs)
        obs, rew, done, info = env.step(actions)
        rew_total += rew

    return info['is_success']


def save_heatmap(means, mass_sweep, friction_sweep, output_path, file_name, show, exp_type):
    fig = plt.figure()
    plt.imshow((means.T), interpolation='nearest', cmap='seismic', aspect='equal', vmin=0, vmax=0.5)
    plt.title(file_name)
    plt.yticks(ticks=(np.arange(len(mass_sweep))), labels=['{:0.2f}'.format(x) for x in mass_sweep])
    plt.ylabel('Mass coef')
    plt.xticks(ticks=(np.arange(len(friction_sweep))), labels=['{:0.2f}'.format(x) for x in friction_sweep])
    plt.xlabel('Friction coef')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('{}/{}_{}.png'.format(output_path, file_name, 'transfer_heatmap'))
    if show:
        plt.show()
    plt.close(fig)


def run_transfer_tests(output_path, env, env_id, model, num_rollouts):
    global_results_list = []
    if env_id == 'MAFetchSlideEnv':
        results_list = []
        for j in range(num_rollouts):
            results_list.append(run_rollout(env, model))

        with open(os.path.join(output_path, 'slide_base'), 'wb') as (file):
            np.savetxt(file, results_list)
        print('average success rate for base is {}'.format(np.mean(results_list)))
        for i, mass in enumerate(slide_mass_sweep):
            for j, fric in enumerate(slide_friction_sweep):
                results_list = []
                set_mass(env.env.env, mass)
                set_friction(env.env.env, fric)
                for j in range(num_rollouts):
                    results_list.append(run_rollout(env, model))

                print('average success rate for m {}, f {} is {}'.format(mass, fric, np.mean(results_list)))
                global_results_list.append(np.mean(results_list))
                with open(os.path.join(output_path, 'slide_f{}_m{}.txt').format(fric, mass), 'wb') as (file):
                    np.savetxt(file, results_list[1:])

    else:
        if env_id == 'MAFetchPushEnv':
            results_list = []
            for j in range(num_rollouts):
                results_list.append(run_rollout(env, model))

            with open(os.path.join(output_path, 'push_base'), 'wb') as (file):
                np.savetxt(file, results_list)
            print('average success rate for base is'.format(np.mean(results_list)))
            for i, mass in enumerate(push_mass_sweep):
                for j, fric in enumerate(push_friction_sweep):
                    results_list = []
                    set_mass(env.env.env, mass)
                    set_friction(env.env.env, fric)
                    for j in range(num_rollouts):
                        results_list.append(run_rollout(env, model))

                    print('average success rate for m {}, f {} is {}'.format(mass, fric, np.mean(results_list)))
                    global_results_list.append(np.mean(results_list))
                    with open(os.path.join(output_path, 'push_f{}_m{}.txt').format(fric, mass), 'wb') as (file):
                        np.savetxt(file, results_list)

        reward_means = np.array(global_results_list).reshape(num_samples, num_samples)
        if env_id == 'MAFetchSlideEnv':
            save_heatmap(reward_means, slide_mass_sweep, slide_friction_sweep, output_path, 'slide', False, 'slide')
        elif env_id == 'MAFetchPushEnv':
            save_heatmap(reward_means, push_mass_sweep, push_friction_sweep, output_path, 'push', False, 'push')
# okay decompiling transfer_tests.cpython-36.pyc
