from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ray.rllib.utils import try_import_tf

from envs.inverted_pendulum_env import PendulumEnv
from models.GTrXL import make_GRU_TrXL

tf = try_import_tf()

def make_model(seq_length, num_outputs, num_layers, attn_dim, num_heads,
               head_dim, ff_hidden_dim):

    return tf.keras.Sequential((
        make_GRU_TrXL(seq_length, num_layers, attn_dim, num_heads,
                            head_dim, ff_hidden_dim),
        tf.keras.layers.Dense(num_outputs),
    ))


def train_loss(delta_state, outputs):
    tf.keras.losses.MSE(
        delta_state,
        outputs
    )


def random_policy():
    return np.random.uniform(-2, 2)


def generate_random_data(env, train_batch_size, seq_len):
    training_data = np.zeros((seq_len, train_batch_size, env.observation_space.shape[0] + env.action_space.shape[0]))
    batch_step = 0
    # get the initial obs
    if env.step_num > env.horizon:
        obs = env.reset()
    else:
        obs = env._get_obs() / env.obs_norm
    # Now run the training loop
    while batch_step < train_batch_size:
        seq_steps = 0
        while seq_steps < seq_len:
            action = random_policy()
            # TODO(@evinitsky) switch this to predicting next state
            training_data[seq_steps, batch_step] = np.concatenate((obs, [action]))
            seq_steps += 1
            obs, rew, done, info = env.step([action])
            if done:
                obs = env.reset()
        batch_step += 1
    return training_data

def shooting_policy():
    pass


def train_bit_shift(seq_length, num_iterations, print_every_n):

    env = PendulumEnv({'horizon': 200})
    env.reset()

    optimizer = tf.keras.optimizers.Adam(1e-3)


    model = make_model(
        seq_length,
        num_outputs=3,
        num_layers=1,
        attn_dim=10,
        num_heads=5,
        head_dim=20,
        ff_hidden_dim=20,
    )

    shift = 10
    train_batch = 10
    test_batch = 100
    data_gen = generate_random_data(env, train_batch, seq_length)
    test_gen = generate_random_data(env, test_batch, seq_length)

    @tf.function
    def update_step(inputs, targets):
        loss_fn = lambda: train_loss(targets, model(inputs))
        var_fn = lambda: model.trainable_variables
        optimizer.minimize(loss_fn, var_fn)

    import ipdb; ipdb.set_trace()
    for i, (inputs, targets) in zip(range(num_iterations), data_gen):
        update_step(
            tf.convert_to_tensor(inputs), tf.convert_to_tensor(targets))

        if i % print_every_n == 0:
            test_inputs, test_targets = next(test_gen)
            print(i, train_loss(test_targets, model(test_inputs)))


if __name__ == "__main__":
    tf.enable_eager_execution()
    num_inputs = 4
    train_bit_shift(
        seq_length=20 * num_inputs,
        num_iterations=20000,
        print_every_n=200,
    )