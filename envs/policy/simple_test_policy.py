import numpy as np
import itertools
import logging
from envs.policy.policy import Policy
from envs.utils.action import ActionRot, ActionXY
from envs.utils.state import ObservableState, FullState

class CADRL(Policy):
    def __init__(self, config):
        super().__init__()
        self.name = 'CADRL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

        self.configure(config)
        # TODO(@evinitsky) figure out a better way to construct the action space
        self.build_action_space(0.0)

    def configure(self, config):
        self.set_common_parameters(config)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        logging.info('Policy: CADRL without occupancy map')

    def set_common_parameters(self, config):
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        assert len(state.human_states) == 1
        state = state.self_state + state.human_states[0]
        state = self.rotate(state.unsqueeze(0)).squeeze(axis=0)
        return state

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = np.arctan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = np.linalg.norm(np.concatenate([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * np.cos(rot) + state[:, 3] * np.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * np.cos(rot) - state[:, 2] * np.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = np.zeros_like(v_pref)
        vx1 = (state[:, 11] * np.cos(rot) + state[:, 12] * np.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * np.cos(rot) - state[:, 11] * np.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * np.cos(rot) + (state[:, 10] - state[:, 1]) * np.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * np.cos(rot) - (state[:, 9] - state[:, 0]) * np.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = np.linalg.norm(np.concatenate([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = np.concatenate([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
