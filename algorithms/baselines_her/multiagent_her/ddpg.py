# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/eugenevinitsky/Desktop/Research/Code/adversarial_sim2real/algorithms/baselines_her/multiagent_her/ddpg.py
# Compiled at: 2020-05-22 16:29:58
# Size of source mod 2**32: 22297 bytes
from collections import OrderedDict
import numpy as np, tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines import logger
from algorithms.baselines_her.multiagent_her.util import import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major
from algorithms.baselines_her.multiagent_her.normalizer import Normalizer
from algorithms.baselines_her.multiagent_her.replay_buffer import ReplayBuffer
from algorithms.baselines_her.multiagent_her.mpi_adam import MpiAdam
from baselines.common import tf_util

def dims_to_shapes(input_dims):
    return {key:tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):

    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size, Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T, rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return, bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight, sample_transitions, gamma, reuse=False, name='agent', **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss
        """
        global DEMO_BUFFER
        if self.clip_return is None:
            self.clip_return = np.inf
        self.create_actor_critic = import_function(self.network_class)
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                pass
            else:
                stage_shapes[key] = (
                 *(None, ), *input_shapes[key])

        for key in ('o', 'g'):
            stage_shapes[key + '_2'] = stage_shapes[key]

        stage_shapes['r'] = (None, )
        self.stage_shapes = stage_shapes
        self.scope = self.scope + name
        self.name = name
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
              shapes=(list(self.stage_shapes.values())))
            self.buffer_ph_tf = [tf.placeholder((tf.float32), shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)
        buffer_shapes = {key:(self.T - 1 if key != 'o' else self.T, *input_shapes[key]) for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        if self.name != 'agent':
            buffer_shapes = {key.replace('agent', self.name):val for key, val in buffer_shapes.items()}
        buffer_size = self.buffer_size // self.rollout_batch_size * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)
        DEMO_BUFFER = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=(-self.max_u), high=(self.max_u), size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = (g.reshape)(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return (o, g)

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return (actions, None, None, None)

    def get_actions(self, o, ag, g, noise_eps=0.0, random_eps=0.0, use_target_net=False, compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        vals = [
         policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        feed = {policy.o_tf: o.reshape(-1, self.dimo), 
         policy.g_tf: g.reshape(-1, self.dimg), 
         policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=(np.float32))}
        ret = self.sess.run(vals, feed_dict=feed)
        u = ret[0]
        noise = noise_eps * self.max_u * (np.random.randn)(*u.shape)
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def init_demo_buffer(self, demoDataFile, update_stats=True):
        demoData = np.load(demoDataFile)
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T - 1, 1, self.input_dims[('info_' + key)]), np.float32) for key in info_keys]
        demo_data_obs = demoData['obs']
        demo_data_acs = demoData['acs']
        demo_data_info = demoData['info']
        for epsd in range(self.num_demo):
            obs, acts, goals, achieved_goals = ([], [], [], [])
            i = 0
            for transition in range(self.T - 1):
                obs.append([demo_data_obs[epsd][transition].get('observation')])
                acts.append([demo_data_acs[epsd][transition]])
                goals.append([demo_data_obs[epsd][transition].get('desired_goal')])
                achieved_goals.append([demo_data_obs[epsd][transition].get('achieved_goal')])
                for idx, key in enumerate(info_keys):
                    info_values[idx][(transition, i)] = demo_data_info[epsd][transition][key]

            obs.append([demo_data_obs[epsd][(self.T - 1)].get('observation')])
            achieved_goals.append([demo_data_obs[epsd][(self.T - 1)].get('achieved_goal')])
            episode = dict(o=obs, u=acts,
              g=goals,
              ag=achieved_goals)
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            DEMO_BUFFER.store_episode(episode)
            logger.debug('Demo buffer size currently ', DEMO_BUFFER.get_current_size())
            if update_stats:
                episode['o_2'] = episode['o'][:, 1:, :]
                episode['ag_2'] = episode['ag'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions)
                o, g, ag = transitions['o'], transitions['g'], transitions['ag']
                transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
                self.o_stats.update(transitions['o'])
                self.g_stats.update(transitions['g'])
                self.o_stats.recompute_stats()
                self.g_stats.recompute_stats()
            episode.clear()

        logger.info('Demo buffer size: ', DEMO_BUFFER.get_current_size())

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)
        if update_stats:
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)
            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
         self.Q_loss_tf,
         self.main.Q_pi_tf,
         self.Q_grad_tf,
         self.pi_grad_tf])
        return (
         critic_loss, actor_loss, Q_grad, pi_grad)

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        if self.bc_loss:
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
            transitions_demo = DEMO_BUFFER.sample(self.demo_batch_size)
            for k, values in transitions_demo.items():
                rolloutV = transitions[k].tolist()
                for v in values:
                    rolloutV.append(v.tolist())

                transitions[k] = np.array(rolloutV)

        else:
            transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        elif not len(self.buffer_ph_tf) == len(batch):
            raise AssertionError
        self.sess.run((self.stage_op), feed_dict=(dict(zip(self.buffer_ph_tf, batch))))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return (critic_loss, actor_loss)

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection((tf.GraphKeys.TRAINABLE_VARIABLES), scope=(self.scope + '/' + scope))
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection((tf.GraphKeys.GLOBAL_VARIABLES), scope=(self.scope + '/' + scope))
        return res

    def _create_network(self, reuse=False):
        logger.info('Creating a DDPG agent with action space %d x %s...' % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        with tf.variable_scope('o_stats') as (vs):
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer((self.dimo), (self.norm_eps), (self.norm_clip), sess=(self.sess))
        with tf.variable_scope('g_stats') as (vs):
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer((self.dimg), (self.norm_eps), (self.norm_clip), sess=(self.sess))
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis=0)
        with tf.variable_scope('main') as (vs):
            if reuse:
                vs.reuse_variables()
            self.main = (self.create_actor_critic)(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as (vs):
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = (self.create_actor_critic)(
 target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        if not len(self._vars('main')) == len(self._vars('target')):
            raise AssertionError
        else:
            target_Q_pi_tf = self.target.Q_pi_tf
            clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else np.inf)
            target_tf = (tf.clip_by_value)(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
            self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
            if self.bc_loss == 1:
                if self.q_filter == 1:
                    maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [-1])
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask((tf.boolean_mask(self.main.pi_tf, mask)), maskMain, axis=0) - tf.boolean_mask((tf.boolean_mask(batch_tf['u'], mask)), maskMain, axis=0)))
                    self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
                    self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf
            elif self.bc_loss == 1:
                if self.q_filter == 0:
                    self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask(self.main.pi_tf, mask) - tf.boolean_mask(batch_tf['u'], mask)))
                    self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
                    self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
                    self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf
            else:
                self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
                self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
            Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
            pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
            assert len(self._vars('main/Q')) == len(Q_grads_tf)
            assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=(self._vars('main/Q')))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=(self._vars('main/pi')))
        self.Q_adam = MpiAdam((self._vars('main/Q')), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam((self._vars('main/pi')), scale_grad_by_procs=False)
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1.0 - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        if prefix != '':
            if not prefix.endswith('/'):
                return [(prefix + '/' + key, val) for key, val in logs]
        return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = [
         '_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
         'main', 'target', 'lock', 'env', 'sample_transitions',
         'stage_shapes', 'create_actor_critic']
        state = {k:v for k, v in self.__dict__.items() if all([subname not in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            state['sample_transitions'] = None
        else:
            (self.__init__)(**state)
            for k, v in state.items():
                if k[-6:] == '_stats':
                    self.__dict__[k] = v

            vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
            assert len(vars) == len(state['tf'])
        node = [tf.assign(var, val) for var, val in zip(vars, state['tf'])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)
# okay decompiling ddpg.cpython-36.pyc
