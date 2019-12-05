import argparse
import configparser
import rospy
import time

from actionflow.msg import frame as Frame_msg, pos as Pos_msg
from geometry_msgs.msg import Twist

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


from envs.policy.policy_factory import policy_factory

FRAME_TOPIC = "robot_pos"
ACTION_TOPIC = '/cmd_vel'

class RobootyVisionControl:
    def __init__(self, rllib_config, checkpoint, show_images):
         # NEEDS TO BE ROBOT AGENT/POLICY ID

        rllib_config['num_workers'] = 0

        # Determine agent and checkpoint
        assert rllib_config['env_config']['run'], "No RL algorithm specified in env config!"
        agent_cls = get_agent_class(rllib_config['env_config']['run'])
        # configure the env
        env_name ='CrowdSim-v0'
        if 'multiagent' in rllib_config and rllib_config['multiagent']['policies']:
            self.robot_id = 'robot'
            register_env(env_name, ma_env_creator)
        else:
            self.robot_id = DEFAULT_POLICY_ID
            register_env(env_name, env_creator)

        # Instantiate the agent
        # create the agent that will be used to compute the actions
        agent = agent_cls(env=env_name, config=rllib_config)
        agent.restore(checkpoint)

        policy_params = configparser.RawConfigParser()
        policy_params.read_string(rllib_config['policy_params'])
        policy = policy_factory[rllib_config['policy']](policy_params)
        self.kinematics = policy.kinematics

        self.policy = agent.workers.local_worker().policy_map[self.robot_id]

        self.prev_state = policy.get_initial_state()
        self.use_lstm = len(self.prev_state) > 0
        self.prev_action = policy.action_space.sample()
        self.prev_reward = 0 

        self._last_frame = None

        self.init_get_action(rllib_config, checkpoint, show_images)
        self.__frame_sub = rospy.Subscriber(FRAME_TOPIC, Frame_msg, self.store_frame, queue_size=1)
        self.__agent_action_pub = rospy.Publisher(ACTION_TOPIC, Twist, queue_size=1)

    def store_frame(self, data):
        self._last_frame = data

    def publish_action(self, action):
        assert self.kinematics == 'holonomic':
        vx, vy = action
        vel_msg = Twist()
        vel_msg.linear.x = vx
        vel_msg.linear.y = vy
        __agent_action_pub.publish(vel_msg)
        return vel_msg

    def compute_robot_action(self):
        # NEED TO CONVERT self.last_frame TO OBS VARIABLE
        ros_obs = np.zeros(p.observation_space) # NEEDS TO MATCH OBS SIZE
        
        if self.use_lstm:
            a_action, self.prev_state, _ = agent.compute_action(
                ros_obs,
                state=self.prev_state
                prev_action=self.prev_action,
                prev_reward=self.prev_reward,
                policy_id=self.robot_id)
        else:
            a_action = agent.compute_action(
                ros_obs,
                prev_action=self.prev_action,
                prev_reward=self.prev_reward,
                policy_id=self.robot_id)
        
        a_action = _flatten_action(a_action)  # tuple actions
        self.prev_action = a_action
        self.prev_state = ros_obs
        return self.prev_action

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser = replay_parser(parser)
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    rllib_config, checkpoint = get_config(args)

    ray.init(num_cpus=args.num_cpus)

    robooty_controller = RobootyVisionControl(rllib_config, checkpoint)

    while(True):
        action = robooty_controller.compute_robot_action()
        self.publish_action(action)
        

if __name__ == '__main__':
    main()