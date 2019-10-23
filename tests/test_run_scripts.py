import os
import unittest

from ray.tune import run as run_tune

from utils.parsers import env_parser, init_parser
from run_scripts.random_rollout import run as run_random
from run_scripts.test_rllib_script import setup_exps

class TestRollout(unittest.TestCase):

    def test_rollout_random(self):
        parser = init_parser()
        args = env_parser(parser).parse_args()

        with open(args.env_config, 'r') as file:
            env_config = file.read()

        with open(args.policy_config, 'r') as file:
            policy_config = file.read()

        passed_config = {'env_config': env_config, 'policy_config': policy_config,
                         'policy': args.policy, 'show_images': args.show_images,
                         'train_on_images': True}
        run_random(passed_config)
        passed_config = {'env_config': env_config, 'policy_config': policy_config,
                         'policy': args.policy, 'show_images': args.show_images,
                         'train_on_images': False}
        run_random(passed_config)

    def test_run_script(self):
        alg_run, config, exp_dict, args = setup_exps()

        with open(args.env_config, 'r') as file:
            env_config = file.read()

        with open(args.policy_config, 'r') as file:
            policy_config = file.read()

        # Run it without images
        exp_dict['config']['train_batch_size'] = 200
        exp_dict['stop']['training_iteration'] = 1
        exp_dict['config']['env_config'] = {'policy': args.policy, 'show_images': False,
                                            'train_on_images': False,
                                            'env_config': env_config, 'policy_config': policy_config}
        run_tune(**exp_dict)
        # Run it with images
        exp_dict['config']['env_config'] = {'policy': args.policy, 'show_images': False,
                                            'train_on_images': True,
                                            'env_config': env_config, 'policy_config': policy_config}
        run_tune(**exp_dict)


if __name__ == '__main__':
    unittest.main()
