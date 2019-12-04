from envs.utils.agent import Agent
from envs.utils.state import JointState
import rospy
from geometry_msgs.msg import TwistStamped, Twist, Pose
import numpy as np



class Robot(Agent):
    def __init__(self, config, section):
        self.__agent_action_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def step(self, action):
        # Publishing action

        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            vx, vy = action
            self.vx = vx
            self.vy = vy
        else:
            r, v = action
            self.theta = (self.theta + r) % (2 * np.pi)
            self.vx = v * np.cos(self.theta)
            self.vy = v * np.sin(self.theta)

        action = self.get_cmd_vel_()
        self.__agent_action_pub.publish(action)
