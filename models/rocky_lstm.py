"""The purpose of this is to allow you to get the actions of multiple agents, but only pass gradients
through the one agent that is active at the time"""

import gym
from gym.spaces import Discrete
import numpy as np
import random
import argparse

import ray
from ray import tune
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class RockyLSTM(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        super(RockyLSTM, self).__init__(obs_space, action_space, num_outputs,
                                        model_config, name)
        self.cell_size = model_config['lstm_cell_size']

        # policy
        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=input_layer,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute actions
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            bias_initializer='zeros',
            name="logits")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

        # value fn
        v_lstm_out, v_state_h, v_state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=input_layer,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(v_lstm_out)

        # Create the RNN model
        self.value_fn = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[values, v_state_h, v_state_c])
        self.register_variables(self.value_fn.variables)
        self.value_fn.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().
        You should implement forward_rnn() in your subclass."""
        output, new_state = self.forward_rnn(
            add_time_dimension(input_dict["obs"], seq_lens), state,
            seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    @override(RecurrentTFModelV2)
    def forward_rnn(self, input_dict, state, seq_lens, prev_action=None):
        # by subclassing recurrent_tf_modelv2, forward_rnn receives
        # inputs that are B x T x features
        actions, h, c = self.rnn_model([input_dict, seq_lens] + state[0:2])
        self._value_out, v_h, v_c = self.rnn_model([input_dict, seq_lens] + state[2:4])
        return actions, [h, c, v_h, v_c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
