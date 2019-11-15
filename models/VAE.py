import numpy as np
import json
import tensorflow as tf
import os
import sys


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


# TODO(@evinitsky) switch this to a regular VAE

class ConvVAE(object):
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5,
                 kernel_size=4, is_training=False, reuse=False,
                 gpu_mode=False, top_percent=1.0):
        """

        :param z_size: (int)
            The size of the bottleneck layer
        :param batch_size: (int)
            The size of a minibatch
        :param learning_rate: (float)
            The learning rate for SGD
        :param kl_tolerance: (float)
            Not used, should be removed
        :param kernel_size: (int)
            The size of the kernel in each layer
        :param is_training: (bool)
            Not used, should be removed
        :param reuse:
        :param gpu_mode:
        :param top_percent: (float)
            what fraction of the loss tensor we should take
        """
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.kernel_size = kernel_size
        self.reuse = reuse
        self.top_percent = top_percent
        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self._build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()
        self._init_session()

    def extract_activations_and_weights(self, activation_list, weight_list, name):
        pass


    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():

            self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

            activation_list = []
            activation_name_list = []

            # Encoder
            h = tf.layers.conv2d(self.x, 32, self.kernel_size, strides=2, activation=tf.nn.relu, name="enc_conv1")
            activation_list.append(h)
            activation_name_list.append("enc_conv1")
            h = tf.layers.conv2d(h, 64, self.kernel_size, strides=2, activation=tf.nn.relu, name="enc_conv2")
            activation_list.append(h)
            activation_name_list.append("enc_conv2")
            h = tf.layers.conv2d(h, 128, self.kernel_size, strides=2, activation=tf.nn.relu, name="enc_conv3")
            activation_list.append(h)
            activation_name_list.append("enc_conv3")
            h = tf.layers.conv2d(h, 256, self.kernel_size, strides=2, activation=tf.nn.relu, name="enc_conv4")
            activation_list.append(h)
            activation_name_list.append("enc_conv4")

            # this is the size of the image after all the convolutional layers
            if self.kernel_size == 4:
                final_shape = 4
            elif self.kernel_size == 3:
                final_shape = 9
            elif self.kernel_size == 2:
                final_shape = 16
            else:
                sys.exit('Only kernel sizes of 2, 3 and 4 are currently supported')
            h = tf.reshape(h, [-1, final_shape * 256])

            # VAE
            self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
            # activation_list.append(self.mu)
            # activation_name_list.append("enc_fc_mu")
            # self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
            # activation_list.append(self.logvar)
            # activation_name_list.append("enc_fc_log_var")
            # self.sigma = tf.exp(self.logvar / 2.0)
            # self.epsilon = tf.random_normal([self.batch_size, self.z_size])
            # self.z = self.mu + self.sigma * self.epsilon

            # Decoder
            h = tf.layers.dense(self.mu, final_shape * 256, name="dec_fc")
            activation_list.append(h)
            activation_name_list.append("dec_fc")
            h = tf.reshape(h, [-1, 1, 1, final_shape * 256])
            # TODO(@evinitsky) how should we think about what the strides and kernels should be here?
            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            activation_list.append(h)
            activation_name_list.append("dec_deconv1")
            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            activation_list.append(h)
            activation_name_list.append("dec_deconv2")
            h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            activation_list.append(h)
            activation_name_list.append("dec_deconv3")
            self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
            activation_list.append(h)
            activation_name_list.append("dec_deconv4")

            self.activation_list = activation_list
            self.activation_name_list = activation_name_list

            # train ops
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                eps = 1e-6  # avoid taking log of zero

                # reconstruction loss. Not that we hard negative mine the top_percent of the loss
                square_diff = tf.square(self.x - self.y)
                shape = square_diff.get_shape().as_list()
                dim = np.prod(shape[1:])
                diff_flat = tf.reshape(square_diff, [-1, dim])
                top_k = int(self.top_percent * dim)
                top_loss, _ = tf.math.top_k(diff_flat, top_k)
                top_loss = tf.reduce_sum(top_loss, reduction_indices=[1])
                self.r_loss = tf.reduce_mean(top_loss)

                # augmented kl loss per dim
                # self.kl_loss = - 0.5 * tf.reduce_sum(
                #     (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
                #     reduction_indices=1
                # )
                # self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                # self.kl_loss = tf.reduce_mean(self.kl_loss)

                self.loss = self.r_loss # + self.kl_loss

                # training
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                self.grads = self.optimizer.compute_gradients(self.loss)  # can potentially clip gradients here.

                self.train_op = self.optimizer.apply_gradients(
                    self.grads, global_step=self.global_step, name='train_step')

            # initialize vars
            self.init = tf.global_variables_initializer()

            t_vars = tf.trainable_variables()
            self.assign_ops = {}
            for var in t_vars:
                # if var.name.startswith('conv_vae'):
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def encode(self, x):
        return self.sess.run(self.mu, feed_dict={self.x: x})
    #
    # def encode_mu_logvar(self, x):
    #     (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
    #     return mu, logvar

    def decode(self, z):
        return self.sess.run(self.y, feed_dict={self.mu: z})

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                # if var.name.startswith('conv_vae'):
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            # rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up
        return rparam

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                # if var.name.startswith('conv_vae'):
                pshape = tuple(var.get_shape().as_list())
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op, pl = self.assign_ops[var]
                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                idx += 1

    def load_json(self, jsonfile='vae.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def save_json(self, jsonfile='vae.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)