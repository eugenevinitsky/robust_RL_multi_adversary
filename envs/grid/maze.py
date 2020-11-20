# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Implements single-agent manually generated Maze environments.
Humans provide a bit map to describe the position of walls, the starting
location of the agent, and the goal location.
"""
from gym.spaces import Box, Discrete
from gym_minigrid.minigrid import *
from envs.grid.register import register
from ray.rllib.env import MultiAgentEnv
import numpy as np


class MazeEnv(MiniGridEnv):
  """Single-agent maze environment specified via a bit map."""

  def __init__(self, agent_view_size=5, minigrid_mode=True, max_steps=None,
               bit_map=None, start_pos=None, goal_pos=None, size=15):
    default_agent_start_x = 7
    default_agent_start_y = 1
    default_goal_start_x = 7
    default_goal_start_y = 13
    self.grid_size = size
    self.start_pos = np.array(
      [default_agent_start_x,
       default_agent_start_y]) if start_pos is None else start_pos
    self.goal_pos = (
      default_goal_start_x,
      default_goal_start_y) if goal_pos is None else goal_pos

    if max_steps is None:
      max_steps = 2 * size * size

    if bit_map is not None:
      bit_map = np.array(bit_map)
      if bit_map.shape != (size - 2, size - 2):
        print('Error! Bit map shape does not match size. Using default maze.')
        bit_map = None

    if bit_map is None:
      self.bit_map = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
      ])
    else:
      self.bit_map = bit_map

    super().__init__(
      grid_size=size,
      agent_view_size=agent_view_size,
      max_steps=max_steps,
    )

  def _gen_grid(self, width, height):
    # Create an empty grid
    self.grid = Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    # Goal
    self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

    # Agent
    self.agent_pos = self.start_pos
    self.agent_dir = 0

    # Walls
    for x in range(self.bit_map.shape[0]):
      for y in range(self.bit_map.shape[1]):
        if self.bit_map[y, x]:
          # Add an offset of 1 for the outer walls
          self.put_obj(Wall(), x + 1, y + 1)
    self.mission = (
      "get to the goal green square"
    )


class MultiMazeEnv(MultiAgentEnv, MiniGridEnv):
  """Single-agent maze environment specified via a bit map."""

  def __init__(self, agent_view_size=5, minigrid_mode=True, max_steps=None,
               bit_map=None, start_pos=None, goal_pos=None, size=15, config=None):
    default_agent_start_x = 7
    default_agent_start_y = 1
    default_goal_start_x = 7
    default_goal_start_y = 13
    self.grid_size = size
    self.start_pos = np.array(
      [default_agent_start_x,
       default_agent_start_y]) if start_pos is None else start_pos
    self.goal_pos = (
      default_goal_start_x,
      default_goal_start_y) if goal_pos is None else goal_pos

    if max_steps is None:
      max_steps = 2 * size * size

    if bit_map is not None:
      bit_map = np.array(bit_map)
      if bit_map.shape != (size - 2, size - 2):
        print('Error! Bit map shape does not match size. Using default maze.')
        bit_map = None

    if bit_map is None:
      self.bit_map = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
      ])
    else:
      self.bit_map = bit_map

    # TODO(@evinitsky) refactor to be less silly
    self.adversary_range = config["num_adv_strengths"] * config["advs_per_strength"]
    self.curr_adversary = 0
    super().__init__(
      grid_size=size,
      agent_view_size=agent_view_size,
      max_steps=max_steps,
    )

  def _gen_grid(self, width, height):
    # Create an empty grid
    self.grid = Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    # Goal
    self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

    # Agent
    self.agent_pos = self.start_pos
    self.agent_dir = 0

    # Walls
    for x in range(self.bit_map.shape[0]):
      for y in range(self.bit_map.shape[1]):
        if self.bit_map[y, x]:
          # Add an offset of 1 for the outer walls
          self.put_obj(Wall(), x + 1, y + 1)
    self.mission = (
      "get to the goal green square"
    )

  # @property
  # def action_space(self):
  #   return Discrete(3)
    self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(low=0, high=255, shape=(75,))})

  @property
  def adv_observation_space(self):
    return gym.spaces.Box(low=-np.infty, high=np.infty, shape=(10,))

  @property
  def adv_action_space(self):
    low = np.zeros(2)
    high = (self.grid_size + 1) * np.ones(2)
    return gym.spaces.Box(low=np.array(low), high=np.array(high))

  def step(self, action_dict, custom_strategy=None):
    action = action_dict['agent']
    self.step_count += 1

    reward = 0
    done = False

    # Get the position in front of the agent
    fwd_pos = self.front_pos

    # Get the contents of the cell in front of the agent
    fwd_cell = self.grid.get(*fwd_pos)

    # Rotate left
    if action == self.actions.left:
      self.agent_dir -= 1
      if self.agent_dir < 0:
        self.agent_dir += 4

    # Rotate right
    elif action == self.actions.right:
      self.agent_dir = (self.agent_dir + 1) % 4

    # Move forward
    elif action == self.actions.forward:
      if fwd_cell == None or fwd_cell.can_overlap():
        self.agent_pos = fwd_pos
      if fwd_cell != None and fwd_cell.type == 'goal':
        done = True
        reward = self._reward()
      if fwd_cell != None and fwd_cell.type == 'lava':
        done = True

    # Pick up an object
    elif action == self.actions.pickup:
      if fwd_cell and fwd_cell.can_pickup():
        if self.carrying is None:
          self.carrying = fwd_cell
          self.carrying.cur_pos = np.array([-1, -1])
          self.grid.set(*fwd_pos, None)

    # Drop an object
    elif action == self.actions.drop:
      if not fwd_cell and self.carrying:
        self.grid.set(*fwd_pos, self.carrying)
        self.carrying.cur_pos = fwd_pos
        self.carrying = None

    # Toggle/activate an object
    elif action == self.actions.toggle:
      if fwd_cell:
        fwd_cell.toggle(self, fwd_pos)

    # Done action (not used by default)
    elif action == self.actions.done:
      pass

    else:
      assert False, "unknown action"

    if self.step_count >= self.max_steps:
      done = True

    obs = self.gen_obs()

    self.step_num += 1
    # if self.step_num == 1:
    #   if self.adversary_range > 0:
    #     adv_action = action_dict['adversary{}'.format(self.curr_adversary)]
    #     while self.bit_map[int(adv_action[0]), int(adv_action[1])]:
    #       adv_action[0] += np.random.randint(low=-1, high=1)
    #       adv_action[1] += np.random.randint(low=-1, high=1)
    #       adv_action = np.clip(adv_action, 0, self.grid_size - 1)
    #       self.put_obj(Goal(), int(adv_action[0]), int(adv_action[1]))

    curr_obs = {'agent': {"image": obs['image'].flatten()}}
    curr_rew = {'agent': reward}
    self.total_rew += reward

    if self.adversary_range > 0:

      # the adversaries get observations on the final steps and on the first step
      if done:
        curr_obs.update({
          'adversary{}'.format(self.curr_adversary): np.ones(10)})
        adv_rew_dict = {'adversary{}'.format(self.curr_adversary): - self.total_rew}
        curr_rew.update(adv_rew_dict)

    done_dict = {'__all__': done}
    # self.temp_render()

    return curr_obs, curr_rew, done_dict, {}

  def reset(self):
    self.step_num = 0
    self.total_rew = 0
    # Current position and direction of the agent
    self.agent_pos = None
    self.agent_dir = None

    # Generate a new random grid at the start of each episode
    # To keep the same grid for each episode, call env.seed() with
    # the same seed before calling env.reset()
    self._gen_grid(self.width, self.height)

    # These fields should be defined by _gen_grid
    assert self.agent_pos is not None
    assert self.agent_dir is not None

    # Check that the agent doesn't overlap with an object
    start_cell = self.grid.get(*self.agent_pos)
    assert start_cell is None or start_cell.can_overlap()

    # Item picked up, being carried, initially nothing
    self.carrying = None

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()
    curr_obs = {'agent': {"image": obs['image'].flatten()}}
    if self.adversary_range > 0:
      curr_obs.update({
        'adversary{}'.format(self.curr_adversary): np.ones(10)
      })
    return curr_obs

  def select_new_adversary(self):
    if self.adversary_range > 0:
      self.curr_adversary = np.random.randint(low=0, high=self.adversary_range)

  def temp_render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
    """
    Render the whole-grid human view
    """

    if close:
      if self.window:
        self.window.close()
      return

    if mode == 'human' and not self.window:
      import gym_minigrid.window
      self.window = gym_minigrid.window.Window('gym_minigrid')
      self.window.show(block=False)

    # Compute which cells are visible to the agent
    _, vis_mask = self.gen_obs_grid()

    # Compute the world coordinates of the bottom-left corner
    # of the agent's view area
    f_vec = self.dir_vec
    r_vec = self.right_vec
    top_left = self.agent_pos + f_vec * (self.agent_view_size - 1) - r_vec * (self.agent_view_size // 2)

    # Mask of which cells to highlight
    highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

    # For each cell in the visibility mask
    for vis_j in range(0, self.agent_view_size):
      for vis_i in range(0, self.agent_view_size):
        # If this cell is not visible, don't highlight it
        if not vis_mask[vis_i, vis_j]:
          continue

        # Compute the world coordinates of this cell
        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

        if abs_i < 0 or abs_i >= self.width:
          continue
        if abs_j < 0 or abs_j >= self.height:
          continue

        # Mark this cell to be highlighted
        highlight_mask[abs_i, abs_j] = True

    # Render the whole grid
    img = self.grid.render(
      tile_size,
      self.agent_pos,
      self.agent_dir,
      highlight_mask=highlight_mask if highlight else None
    )

    if mode == 'human':
      self.window.show_img(img)
      self.window.set_caption(self.mission)

    return img


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname
register(
  env_id='MazeEnv-v0',
  entry_point=module_path + ':MazeEnv'
)
register(
  env_id='MultiMazeEnv-v0',
  entry_point=module_path + ':MultiMazeEnv'
)
