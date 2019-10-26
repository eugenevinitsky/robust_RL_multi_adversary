import argparse
import os


def init_parser():
    return argparse.ArgumentParser('Parse some arguments bruv')


def ray_parser(parser):
    parser.add_argument('--exp_title', type=str, default='test', help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument('--num_iters', type=int, default=350)
    parser.add_argument('--checkpoint_freq', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)

    # TODO: Fix this visualization code
    parser.add_argument('--render', type=str, default=False)
    return parser

def env_parser(parser):
    script_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--env_params', type=str,
                        default=os.path.abspath(os.path.join(script_path, '../configs/env_params.config')))
    parser.add_argument('--policy_params', type=str,
                        default=os.path.abspath(os.path.join(script_path, '../configs/policy_params.config')))
    parser.add_argument('--policy', type=str, default='cadrl')
    parser.add_argument('--train_config', type=str, default=os.path.join(script_path, '../configs/train.config'))
    parser.add_argument("--show_images", action="store_true", default=False, help="Whether to display the observations")
    parser.add_argument('--train_on_images', action='store_true', default=False, help='Whether to train on images')
    return parser