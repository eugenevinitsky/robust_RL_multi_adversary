from envs.policy.linear import Linear
from envs.policy.orca import ORCA

from envs.policy.simple_test_policy import CADRL


def none_policy():
    return None


policy_factory = dict()
policy_factory['cadrl'] = CADRL
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
