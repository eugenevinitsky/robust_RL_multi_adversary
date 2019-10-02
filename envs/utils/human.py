from envs.utils.agent import Agent
from envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def set_goal(self, goal):
        """
        Set goal of human agent
        :param goal: Tuple of (gx,gy) for goal x and y positions
        :return: None
        """
        self.gx = goal[0]
        self.gy = goal[1]
