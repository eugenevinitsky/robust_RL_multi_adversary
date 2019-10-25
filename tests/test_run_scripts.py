import base64
import cloudpickle
from datetime import datetime
import glob
import hashlib
import os
import random
import string
import unittest

import ray
from ray.tune import run as run_tune

from utils.parsers import env_parser, init_parser
from run_scripts.random_rollout import run as run_random, setup_random
from run_scripts.test_rllib_script import setup_exps
from visualize.rollout import run_rollout
from models.models import ConvLSTM

class TestRollout(unittest.TestCase):

    def test_rollout_random(self):
        config = setup_random()

        run_random(config)
        config['show_images'] = True
        run_random(config)

    def test_train_script(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        test_out_dir = os.path.join(script_path, "test_out/" + folder)

        exp_dict, args = setup_exps([])

        # Run it without images
        exp_dict['config']['train_batch_size'] = 128
        exp_dict['local_dir'] = test_out_dir
        exp_dict['stop']['training_iteration'] = 1
        run_tune(**exp_dict)
        self.replay_rollout(test_out_dir)

    def test_train_with_images_script(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        test_out_dir = os.path.join(script_path, "test_out/" + folder)
        # Run it with images
        exp_dict, args = setup_exps(["--train_on_images"])
        exp_dict['config']['train_batch_size'] = 128
        exp_dict['local_dir'] = test_out_dir
        exp_dict['stop']['training_iteration'] = 1
        run_tune(**exp_dict)
        self.replay_rollout(test_out_dir)

    def replay_rollout(self, test_out_dir):
        count = 0
        for sub_folder in glob.iglob(os.path.join(test_out_dir, "test/*")):
            if os.path.isdir(sub_folder):
                count += 1
                config_path = os.path.join(sub_folder, "params.pkl")
                with open(config_path, 'rb') as f:
                    rllib_config = cloudpickle.load(f)

                checkpoint = os.path.join(sub_folder, 'checkpoint_1/checkpoint-1')
                run_rollout(rllib_config, checkpoint, False, os.path.join(test_out_dir, "test_replay_{}.mp4".format(count)))    
        assert count == 1


if __name__ == '__main__':
    ray.init()
    unittest.main()
    ray.shutdown()