"""Runs the training script and computes transfer scores."""

import subprocess

if __name__ == '__main__':
    p1 = subprocess.Popen("mpirun -np 4 python -m run --alg=multiagent_her --env=MAFetchPushEnv "
                          "--num_timesteps=5000000 --num_adv 5 --adv_all_actions "
                          "--adv_strength 0.25 --num_adv_strengths 1 --advs_per_strength 5 "
                          "--return_all_obs".split(' '))
    p1.wait(50)