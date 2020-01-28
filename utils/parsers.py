import argparse
import os


def init_parser():
    return argparse.ArgumentParser('Parse some arguments bruv')


def ray_parser(parser):
    parser.add_argument('--exp_title', type=str, default='test',
                        help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument('--local_mode', action='store_true', help='Set to true if this will '
                                                                  'be run in local mode')
    parser.add_argument('--train_batch_size', type=int, default=10000, help='How many steps go into a training batch')
    parser.add_argument('--num_iters', type=int, default=350)
    parser.add_argument('--checkpoint_freq', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--grid_search', action='store_true', help='If true, grid search hyperparams')
    parser.add_argument('--seed_search', action='store_true', help='If true, sweep seed instead of hyperparameter')

    # TODO: Fix this visualization code
    parser.add_argument('--run_transfer_tests', action='store_true', default=False,
                        help='If true run the transfer tests on the results and upload them to AWS')
    parser.add_argument('--render', type=str, default=False)
    parser.add_argument('--use_lstm', default=False, action='store_true', help='If true, use an LSTM')

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
    parser.add_argument('--change_colors_mode', type=str, default='no_change',
                        help='If mode `every_step`, the colors will be swapped '
                             'at each step. If mode `first_step` the colors will'
                             'be swapped only once')
    parser.add_argument('--friction', action='store_true', default=False,
                        help='If true, all the commands are slightly less than expected and the humans move slower')
    parser.add_argument('--friction_coef', type=float, default=0.4,
                        help='The scaling on the friction')
    parser.add_argument('--chase_robot', action='store_true', default=False,
                        help='If true, then human default goal will be robots next location. '
                             'Else, default human behaviour (circle_crossing)')
    parser.add_argument('--restrict_goal_region', action='store_false', default=True,
                        help='If false, then goals can be generated anywhere within accessible_space. Else they will'
                             ' only be generated in the specified goal_region')
    parser.add_argument('--add_gaussian_noise_state', action='store_true', default=False,
                        help='If true, add gaussian noise to the observed states')
    parser.add_argument('--add_gaussian_noise_action', action='store_true', default=False,
                        help='If true, add gaussian noise to the actions')
    parser.add_argument('--num_adv', type=int, default=2, help='Specifies how many adversaries '
                            'are training in the multi-agent setting')
    parser.add_argument('--human_num', type=int, default=1, help='How many humans are in the training scenario')
    return parser


def ma_env_parser(parser):
    parser.add_argument("--perturb_actions", action="store_true", default=False, help="Add adversary to agent actions")
    parser.add_argument("--perturb_state", action="store_true", default=False, help="Add adversary to agent state")
    parser.add_argument("--kl_diff_weight", type=float, default=0.01,
                        help="How much weight to reward differences in kl between policies")
    parser.add_argument("--kl_diff_target", type=float, default=10.0,
                        help="The desired average kl diff between the policies")
    return parser


def replay_parser(parser):
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--show_images', action="store_true")
    parser.add_argument('--num_rollouts', type=int, default=1)

    return parser
