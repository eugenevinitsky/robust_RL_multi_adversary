import argparse
import os

import numpy as np
import ray

from envs.robotics.fetch.slide import MAFetchSlideEnv
slide_mass_sweep = np.linspace(0.5, 3.5, 20)
slide_friction_sweep = np.linspace(0.01, 0.2, 20)

push_friction_sweep = np.linspace(0.1, 2.0, 20)
push_mass_sweep = np.linspace(0.5, 3.5, 20)


# use this
def set_mass(env, mass):
    bnames = env.sim.model.body_names
    bindex = bnames.index('object0')
    env.sim.model.body_mass[bindex] = mass


def set_friction(env, fric):
    bnames = env.sim.model.geom_names
    bindex = bnames.index('object0')
    env.sim.model.geom_friction[bindex] = [fric, env.sim.model.geom_friction[bindex][1],
                                           env.sim.model.geom_friction[bindex][2]]


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


def run_transfer_tests(output_path, env, env_id, model, num_rollouts):
    if env_id == 'MAFetchSlideEnv':
        for i, mass in enumerate(slide_mass_sweep):
            for j, fric in enumerate(slide_friction_sweep):
                results_list = []
                # It's wrapped in two wrappers
                set_mass(env.env.env, mass)
                set_friction(env.env.env, fric)
                for j in range(num_rollouts):
                    results_list.append(run_rollout(env, model))
                print('average success rate for m {}, f {} is {}'.format(mass, fric, np.mean(results_list)))
                with open(os.path.join(output_path, 'slide_f{}_m{}.txt').format(fric, mass), 'wb') as file:
                    np.savetxt(file, results_list)
    elif env_id == 'MAFetchPushEnv':
        for i, mass in enumerate(push_mass_sweep):
            for j, fric in enumerate(push_friction_sweep):
                results_list = []
                # It's wrapped in two wrappers
                set_mass(env.env.env, mass)
                set_friction(env.env.env, fric)
                for j in range(num_rollouts):
                    results_list.append(run_rollout(env, model))
                print('average success rate for m {}, f {} is {}'.format(mass, fric, np.mean(results_list)))
                with open(os.path.join(output_path, 'push_f{}_m{}.txt').format(fric, mass), 'wb') as file:
                    np.savetxt(file, results_list)
