import gym
from gym.envs.registration import register as gym_register

env_list = []


def register(env_id, entry_point, reward_threshold=0.95):
  """Register a new environment with OpenAI gym based on id."""
  if env_id in env_list:
    del gym.envs.registry.env_specs[id]
  else:
    # Add the environment to the set
    env_list.append(id)

  # Register the environment with OpenAI gym
  gym_register(
      id=env_id, entry_point=entry_point, reward_threshold=reward_threshold)