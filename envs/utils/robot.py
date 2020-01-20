from envs.utils.agent import Agent
from envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section, id=None):
        super().__init__(config, section)
        self.id = 'robot' + str(id)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
