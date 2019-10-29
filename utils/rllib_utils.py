import os
import cloudpickle


def get_config(args):
    config_path = os.path.join(args.result_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.result_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    with open(config_path, 'rb') as f:
        rllib_config = cloudpickle.load(f)

    checkpoint = args.result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num

    return rllib_config, checkpoint