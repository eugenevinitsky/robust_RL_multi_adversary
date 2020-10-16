# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46) 
# [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/eugenevinitsky/Desktop/Research/Code/adversarial_sim2real/algorithms/baselines_her/run.py
# Compiled at: 2020-05-18 20:50:25
# Size of source mod 2**32: 8039 bytes
import sys, re, multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf, numpy as np
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from algorithms.baselines_her.multiagent_her.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from run_scripts.mujoco.run_adv_mujoco import get_parser, get_env_config
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[(-1)]
    _game_envs[env_type].add(env.id)

_game_envs['retro'] = {
 'BubbleBobble-Nes',
 'SuperMarioBros-Nes',
 'TwinBee3PokoPokoDaimaou-Nes',
 'SpaceHarrier-Nes',
 'SonicTheHedgehog-Genesis',
 'Vectorman-Genesis',
 'FinalFight-Snes',
 'SpaceInvaders-Snes'}

def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)
    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, (osp.join(logger.get_dir(), 'videos')), record_video_trigger=(lambda x: x % args.save_video_interval == 0), video_length=(args.save_video_length))
    elif args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    model = learn(env=env, 
     seed=seed, 
     total_timesteps=total_timesteps, 
     num_adv=args.num_adv, **alg_kwargs)
    return (
     model, env)


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    env_type, env_id = get_env_type(args)
    temp_config = {'env_config': {}}
    passed_config = get_env_config(args, temp_config)
    if env_type in {'retro', 'atari'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        else:
            if alg == 'trpo_mpi' or alg == 'multiagent_her':
                env = make_env(env_id, env_type, seed=seed)
            else:
                frame_stack_size = 4
                env = make_vec_env(env_id, env_type, nenv, seed, gamestate=(args.gamestate), reward_scale=(args.reward_scale), config=passed_config)
                env = VecFrameStack(env, frame_stack_size)
    else:
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=1,
          inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)
        flatten_dict_observations = alg not in {'multiagent_her', 'her'}
        env = make_env(env_id, env_type, (args.num_env or 1), seed, reward_scale=(args.reward_scale),
          flatten_dict_observations=flatten_dict_observations,
          config=passed_config)
        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)
        return env


def get_env_type(args):
    env_id = args.env
    if args.env_type is not None:
        return (
         args.env_type, env_id)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[(-1)]
        _game_envs[env_type].add(env.id)

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        if not env_id == 'MAFetchReachEnv':
            if env_id == 'MAFetchPushEnv' or env_id == 'MAFetchSlideEnv':
                env_type = 'robotics'
        else:
            env_type = None
            for g, e in _game_envs.items():
                if env_id in e:
                    env_type = g
                    break

            if ':' in env_id:
                env_type = re.sub(':.*', '', env_id)
            assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())
    return (
     env_type, env_id)


def get_default_network(env_type):
    if env_type in {'retro', 'atari'}:
        return 'cnn'
    return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        alg_module = import_module('.'.join(['algorithms.baselines_her', alg, submodule]))
    except ImportError:
        alg_module = import_module('.'.join(['rl_algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}

    return kwargs


def parse_cmdline_kwargs(args):
    """
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    """

    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k:parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        (logger.configure)(**kwargs)


def main(args):
    arg_parser = common_arg_parser()
    arg_parser = get_parser(arg_parser)
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger((args.log_path), format_strs=[])
    model, env = train(args, extra_args)
    if args.save_path is not None:
        if rank == 0:
            save_path = osp.expanduser(args.save_path)
            model.save(save_path)
    if args.play:
        logger.log('Running trained model')
        obs = env.reset()
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1, ))
        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while 1:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)
            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()
    return model


if __name__ == '__main__':
    main(sys.argv)
# okay decompiling run.cpython-37.pyc
