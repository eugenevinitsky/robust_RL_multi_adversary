import os
# from ray import cloudpickle
import pickle

def get_config(args):
    config_path = os.path.join(args.result_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.result_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    with open(config_path, 'rb') as f:
        rllib_config = pickle.load(f)

    checkpoint = args.result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num

    return rllib_config, checkpoint


def get_config_from_path(config_path, checkpoint_num):
    config_path = os.path.join(config_path, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_path, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    with open(config_path, 'rb') as f:
        rllib_config = pickle.load(f)

    checkpoint_path = os.path.dirname(config_path) + '/checkpoint_' + checkpoint_num
    checkpoint_path = checkpoint_path + '/checkpoint-' + checkpoint_num

    return rllib_config, checkpoint_path