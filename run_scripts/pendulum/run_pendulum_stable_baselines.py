import gym

import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import TD3, PPO2, ACKTR
from stable_baselines.results_plotter import load_results, ts2xy

from utils.pendulum_env_creator import pendulum_env_creator

num_adv = 5
horizon = 200
base_vector = np.array([0, 0, -1])
scale_factors = np.linspace(start=0, stop=1, num=num_adv)
state_weights = []
for scale_factor in scale_factors:
    state_weights.append(base_vector * scale_factor)

env_config = {'horizon': horizon, 'num_adversaries': num_adv, 'model_based': True, 'adversary_strength': 0.5,
              'guess_adv': False, 'guess_next_state': False, 'num_concat_states': 1,
              'adversary_type': 'friction', 'weights': state_weights}

gym.register(
    id='ModelBasedPendulum-v0',
    entry_point='envs.inverted_pendulum_env:ModelBasedPendulumEnv',
    max_episode_steps=horizon,
    kwargs={'config': env_config}
)

# multiprocess environment
n_cpus = 2
env = make_vec_env('ModelBasedPendulum-v0', n_envs=n_cpus)
import time
curr_time = time.time()
model = PPO2(policy="MlpLstmPolicy", env=env, learning_rate=5e-3, n_steps=512, nminibatches=2,
             tensorboard_log="/Users/eugenevinitsky/tb_logs").learn(total_timesteps=100000)
print('run time is ', time.time() - curr_time)
model.save("pendulum")

del model # remove to demonstrate saving and loading

model = TD3.load("pendulum")

# Enjoy trained agent
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
