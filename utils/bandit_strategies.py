import argparse
from datetime import datetime
import numpy as np
import os
import pytz
import sys

from envs.multiarm_bandit import MultiarmBandit
from run_scripts.bandit.run_multiarm_bandit import setup_exps
from visualize.pendulum.transfer_tests import reset_env, make_bandit_transfer_list
from visualize.pendulum.run_rollout import run_non_rl_rollout
from parsers import init_parser
import matplotlib.pyplot as plt

class BanditStrategy:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_score_history = [[] for _ in range(self.num_arms)]

    def get_arm(self, env, step_num):
        return 0

    def add_reward(self, arm_choice, reward):
        self.arm_score_history[arm_choice].append(reward)

    def reset(self):
        self.arm_score_history = [[] for _ in range(self.num_arms)]

class OracleStrategy(BanditStrategy):
    def get_arm(self, env, step_num):
        return np.argmax(env.means)

class RandomStrategy(BanditStrategy):
    def get_arm(self, env, step_num):
        return np.random.randint(self.num_arms)

class EpsilonGreedy(BanditStrategy):
    def __init__(self, num_arms, eps=0.1, init_predicted_mean=0.0):
        super(EpsilonGreedy, self).__init__(num_arms)
        self.eps = eps
        self.init_predicted_mean = init_predicted_mean

    def get_arm(self, env, step_num):
        if np.random.rand() < self.eps:
            # print("Random!")
            return np.random.randint(self.num_arms)
        else:
            # print("Best!")
            predicted_means = [np.mean(rew) if len(rew) else self.init_predicted_mean for rew in self.arm_score_history]
            return np.argmax(predicted_means)

class UCB1(BanditStrategy):
    def __init__(self, num_arms, init_predicted_mean=0.0):
        super(UCB1, self).__init__(num_arms)
        self.init_predicted_mean = init_predicted_mean

    def get_arm(self, env, step_num):
        predicted_means = np.array([np.mean(rew) if len(rew) else self.init_predicted_mean for rew in self.arm_score_history])
        ucb = [np.sqrt(2 * np.log(step_num) / (1 + len(rew))) for rew in self.arm_score_history]
        return np.argmax(predicted_means)

class BayesianUCB(BanditStrategy):
    def __init__(self, num_arms, c=1, init_mean=0.0, init_std=0.5, clipping=False, sample=False):
        super(BayesianUCB, self).__init__(num_arms)
        self.init_mean = init_mean
        self.init_std = init_std
        self.c = c
        self.clipping = True
        self.sampling = False

    def get_arm(self, env, step_num):
        means = np.array([np.mean(rew) if len(rew) else self.init_mean for rew in self.arm_score_history])
        stds = np.array([np.std(rew) if len(rew) else self.init_std for rew in self.arm_score_history])
        if self.clipping:
            means = np.clip(means, env.min_mean_reward, env.max_mean_reward)
            stds = np.clip(stds, env.min_std, env.max_std)
        if self.sampling:
            self.sampling = np.random.multivariate_normal(means, np.diag(stds))
        else:
            predicted_means = means + self.c * stds
            return np.argmax(predicted_means)


def make_epsilon_greedy_strategy(epsilon, num_arms):
    
    def epsilon_greedy(env, step_num):
        if self.step_num == 0:
            arm_scores = np.zeros(num_arms)

        assert type(env) is MultiarmBandit
        return np.random.randint(env.num_arms)


def run_bandit_optimal_transfer_tests(strategy, test_name, config, output_file_name, outdir, num_rollouts):
    output_file_path = os.path.join(outdir, output_file_name)
    if not os.path.exists(os.path.dirname(output_file_path)):
        try:
            os.makedirs(os.path.dirname(output_file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    run_list = make_bandit_transfer_list(config['num_arms'])
    transfer_results = []
    for transfer in run_list:
        name, env_modifier = transfer       
        env = MultiarmBandit(config)
        reset_env(env)
        
        if callable(env_modifier):
            env_modifier(env)
        elif len(env_modifier) > 0:
            env.transfer = env_modifier
        
        rewards, step_num = run_non_rl_rollout(env, strategy, num_rollouts)

        with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, test_name),
                'wb') as file:
            np.savetxt(file, rewards, delimiter=', ')

        print('The average reward for task {} is {}'.format(test_name, np.mean(rewards)))
        print('The average step length for task {} is {}'.format(test_name, np.mean(step_num)))

        transfer_results.append((np.mean(rewards), np.std(rewards), np.mean(step_num), np.std(step_num)))
        
    with open('{}/{}_{}_rew.txt'.format(outdir, output_file_name, "mean_sweep"),
              'wb') as file:
        np.save(file, np.array(transfer_results))

    means = np.array(transfer_results)[:,0]
    std_devs = np.array(transfer_results)[:,1]
    if len(means) > 0:
        with open('{}/{}_{}.png'.format(outdir, output_file_name, "transfer_performance"), 'wb') as transfer_robustness:
            fig = plt.figure(figsize=(10, 5))
            plt.bar(np.arange(len(means)), means)
            plt.title("Bandit performance tests")
            plt.xticks(ticks=np.arange(len(means)), labels=[transfer[0] for transfer in run_list])
            plt.xlabel("Bandit test name")
            plt.ylabel("Bandit regret")
            plt.savefig(transfer_robustness)
            plt.close(fig)

def get_strategy(strategy_name):
    if strategy_name == 'oracle':
        strategy = OracleStrategy(args.num_arms)
    elif strategy_name =='random':
        strategy = RandomStrategy(args.num_arms)
    elif strategy_name == 'eps_greedy':
        strategy = EpsilonGreedy(args.num_arms)
    elif strategy_name == 'ucb1':
        strategy = UCB1(args.num_arms)
    elif strategy_name == 'bayesian':
        strategy = BayesianUCB(args.num_arms)
    elif strategy_name == 'bayesian_clip':
        strategy = BayesianUCB(args.num_arms, clipping=True)
    elif strategy_name == 'thompson':
        strategy = BayesianUCB(args.num_arms, sample=True)
    elif strategy_name == 'thompson_clip':
        strategy = BayesianUCB(args.num_arms, sample=True, clipping=True)
    return strategy

if __name__ == "__main__":
    strategies = ['oracle', 'random', 'eps_greedy', 'ucb1', 'bayesian', 'bayesian_clip', 'thompson', 'thompson_clip',]
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    output_path = os.path.expanduser('~/transfer_results/')

    parser = init_parser()
    parser.add_argument('--output_file_name', type=str, default='transfer_out',
                        help='The file name we use to save our results')
    parser.add_argument('--output_dir', type=str, default=output_path,
                        help='')
    parser.add_argument('--num_rollouts', type=int, default=1,
                        help='')
    parser.add_argument('--strategy', type=str, default='oracle', choices=strategies + ['all'],
                        help='')

    exp_dict, args = setup_exps(sys.argv[1:], parser)

    if args.strategy == 'all':
        for strategy_name in strategies:
            strategy = get_strategy(strategy_name)
            output_file_name = "{}/{}".format(args.output_file_name, strategy_name)
            run_bandit_optimal_transfer_tests(strategy, strategy_name, exp_dict['config']['env_config'], output_file_name, args.output_dir, args.num_rollouts)
    else:
        strategy = get_strategy(args.strategy)
        run_bandit_optimal_transfer_tests(strategy, args.strategy, exp_dict['config']['env_config'], args.output_file_name, args.output_dir, args.num_rollouts)





    
