from ray.rllib.models import Model
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.fcnet_v1 import FullyConnectedNetwork
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

class AutoEncoderAdversary(Model):
    """Custom model that adds an imitation loss on top of the policy loss."""

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
            self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                               self.action_space, num_outputs,
                                               options)
        return self.fcnet.outputs, self.fcnet.last_layer

    def custom_loss(self, policy_loss, loss_inputs):
        # create a new input reader per worker
        reader = JsonReader(self.options["custom_options"]["input_files"])
        input_ops = reader.tf_input_ops()

        # define a secondary loss by building a graph copy with weight sharing
        obs = tf.cast(input_ops["obs"], tf.float32)
        logits, _ = self._build_layers_v2({
            "obs": restore_original_dimensions(obs, self.obs_space)
        }, self.num_outputs, self.options)

        # You can also add self-supervised losses easily by referencing tensors
        # created during _build_layers_v2(). For example, an autoencoder-style
        # loss can be added as follows:
        ae_loss = squared_diff(loss_inputs["obs"], Decoder(self.fcnet.last_layer))
        print("FYI: You can also use these tensors: {}, ".format(loss_inputs))

        # compute the IL loss
        action_dist = Categorical(logits, self.options)
        self.policy_loss = policy_loss
        self.imitation_loss = tf.reduce_mean(
            -action_dist.logp(input_ops["actions"]))
        return policy_loss + 10 * self.imitation_loss

    def custom_stats(self):
        return {
            "policy_loss": self.policy_loss,
            "imitation_loss": self.imitation_loss,
        }