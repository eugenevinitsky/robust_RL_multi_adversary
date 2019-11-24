import numpy as np

from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

tf = try_import_tf()


class ConvLSTM(RecurrentTFModelV2):
    """
    fcnet_hiddens = [[layers before lstm], [layers after lstm]]
    
    Arguments:
        TFModelV2 {[type]} -- [description]
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ConvLSTM, self).__init__(obs_space, action_space, num_outputs,
                                    model_config, name)
        self.obs_space = obs_space
        self.num_outputs = num_outputs

        ## Batch x Time x H x W x C
        input_layer = tf.keras.layers.Input(shape=(None,) + obs_space.shape, name="inputs")

        conv_activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        last_layer = input_layer
        for i, (out_size, kernel, stride) in enumerate(filters):
            ## Batch x Time x H x W x C 
            # Time distributed ensures that the conv operates on each image independently
            last_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=conv_activation,
                padding="same",
                name="conv{}".format(i)))(last_layer)
        
        last_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(last_layer)

        # If true we append the actions into the layer after the conv
        self.use_prev_action = model_config["custom_options"].get("use_prev_action")
        if self.use_prev_action:
            actions_layer = tf.keras.layers.Input(shape=(None, action_space.shape), name="agent_actions")
            last_layer = tf.keras.layers.concatenate([last_layer, actions_layer])

        hiddens = model_config["custom_options"].get("fcnet_hiddens") # should be list of lists
        assert type(hiddens) == list
        assert type(hiddens[0]) == list
        assert type(hiddens[1]) == list
        i = 1
        fc_activation = get_activation_fn(model_config.get("fcnet_activation"))
        for size in hiddens[0]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=fc_activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        self.cell_size = model_config['lstm_cell_size']

        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # expects B x T x (H*W*C)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm")(
            inputs=last_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])        
        # output: B x cell_size
        
        last_layer = lstm_out
        for size in hiddens[1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=fc_activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        action = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="action_logits")(last_layer)

        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(last_layer)

        inputs = [input_layer, seq_in, state_in_h, state_in_c]
        if self.use_prev_action:
            inputs.insert(1, actions_layer)
        outputs = [action, values, state_h, state_c]

        self.rnn_model = tf.keras.Model(
                inputs=inputs,
                outputs=outputs)
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().
        You should implement forward_rnn() in your subclass."""
        if self.use_prev_action:
            output, new_state = self.forward_rnn(
                add_time_dimension(input_dict["obs"], seq_lens), state,
                seq_lens, add_time_dimension(input_dict["prev_action"], seq_lens))
        else:
            output, new_state = self.forward_rnn(
                add_time_dimension(input_dict["obs"], seq_lens), state,
                seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    @override(RecurrentTFModelV2)
    def forward_rnn(self, input_dict, state, seq_lens, prev_action=None):
        # by subclassing recurrent_tf_modelv2, forward_rnn receives
        # inputs that are B x T x features
        if prev_action:
            model_out, self._value_out,  h, c = self.rnn_model([input_dict, prev_action, seq_lens] + state)
        else:
            model_out, self._value_out,  h, c = self.rnn_model([input_dict, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])
