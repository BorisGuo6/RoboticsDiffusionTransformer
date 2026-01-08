#!/usr/bin/env python
# -- coding: UTF-8

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import rospy
import torch
# from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PImage
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String
import cv2
from utils import imgmsg_to_cv2
from RoboticsDiffusionTransformer.scripts.franka_model_joint_v import create_model
import cv_bridge


from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output as GripperOutput
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input as GripperInput

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = None
lang_embeddings = None
preload_images = None

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Initialize the model (unchanged)
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate actions (unchanged)
def interpolate_action(args, prev_action, cur_action):
    steps = np.array(args.arm_steps_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_ros_observation(args, ros_operator):
    delay = 1.0 / args.publish_rate  # Use the desired frequency (e.g., from args)
    print_flag = True

    while not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            time.sleep(delay)
            continue
        print_flag = True
        (img_front, img_left, img_right, puppet_arm_right, gripper_state) = result

        # print(f"sync success when get_ros_observation")
        return (img_front, img_left, img_right, puppet_arm_right, gripper_state)


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
            }
        )

    img_front, img_left, img_right, puppet_arm_right, gripper_state = get_ros_observation(args, ros_operator)
    img_front = jpeg_mapping(img_front)
    img_left = jpeg_mapping(img_left)
    img_right = jpeg_mapping(img_right)

    qpos = np.array(list(puppet_arm_right.position[:7]) + [gripper_state.gPO])
    qpos = torch.from_numpy(qpos).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                    config["camera_names"][2]: None,
                },
        }
    )


# RDT inference
def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings

    # print(f"Start inference_thread_fn: t={t}")
    while not rospy.is_shutdown():
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            observation_window[-2]['images'][config['camera_names'][2]],

            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]],
            observation_window[-1]['images'][config['camera_names'][2]]
        ]

        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]

        # get last qpos in shape [14, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")

        # print(f"Model inference time: {time.time() - time1} s")

        # print(f"Finish inference_thread_fn: t={t}")
        return actions

class RosOperator:
    def __init__(self, args):
        self.args = args
        # self.bridge = CvBridge()

        # Initialize deques
        self.img_left_deque = deque(maxlen=2000)
        self.img_right_deque = deque(maxlen=2000)
        self.img_front_deque = deque(maxlen=2000)
        self.img_left_depth_deque = deque(maxlen=2000)
        self.img_right_depth_deque = deque(maxlen=2000)
        self.img_front_depth_deque = deque(maxlen=2000)
        self.puppet_arm_left_deque = deque(maxlen=2000)
        self.puppet_arm_right_deque = deque(maxlen=2000)
        self.gripper_state_deque = deque(maxlen=2000)

        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_thread = None
        self.init_ros()

    def init_ros(self):
        # Publishers
        self.puppet_arm_left_publisher = rospy.Publisher(
            self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(
            self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        
        self.puppet_gripper_publisher = rospy.Publisher(
            self.args.gripper_cmd_topic,GripperOutput, queue_size=10)

        # Subscribers
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=10)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=10)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=10)

        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState,
                         self.puppet_arm_right_callback, queue_size=10)
        
        rospy.Subscriber(
            self.args.gripper_state_topic,
            GripperInput,
            self.gripper_state_callback
        )


    def init_gripper(self):
        """Initialize the Robotiq gripper"""
        command = GripperOutput()
        command.rACT = 1  # Activate gripper
        command.rGTO = 1  # Go to position
        command.rSP = 255  # Speed (255 is maximum)
        command.rFR = 150  # Force (150 is a good default)
        command.rPR = 3
        self.puppet_gripper_publisher.publish(command)
        print("Franka Robotiq gripper initialized!")
        rospy.sleep(1.0)  # Wait for activation

    def puppet_arm_publish(self, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = [float(x) for x in right][:7]

        self.puppet_arm_right_publisher.publish(joint_state_msg)

        gripper_cmd_msg = GripperOutput()
        gripper_cmd_msg.rACT = 1  # Gripper activated
        gripper_cmd_msg.rGTO = 1  # Go to position
        gripper_cmd_msg.rSP = 255  # Speed
        gripper_cmd_msg.rFR = 150  # Force
        gripper_cmd_msg.rPR = int(right[7]) # Position (255 is fully closed)

        self.puppet_gripper_publisher.publish(gripper_cmd_msg)
        print(f"gripper position: {int(right[7])}")


    def puppet_arm_publish_continuous(self, right):
        right_arm = None
        rate = rospy.Rate(self.args.publish_rate)

        while not rospy.is_shutdown():
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
                # print(f"right_arm{right_arm}")
                break
            rate.sleep()

        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0

        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                print("arm publish lock")
                return

            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False

            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True

            right_joint_msg = JointState()
            right_joint_msg.header = Header()
            right_joint_msg.header.stamp = rospy.Time.now()
            right_joint_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            right_joint_msg.position = right_arm[:7]
            # right_joint_msg = String()
            # right_joint_msg.data = str(right_arm)
            self.puppet_arm_right_publisher.publish(right_joint_msg)

            step += 1
            # print(f"puppet_arm_publish_continuous: step {step}")
            rate.sleep()

    # Callback methods (mostly unchanged except for timestamp handling)
    def img_left_callback(self, msg):
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        self.puppet_arm_right_deque.append(msg)

    def gripper_state_callback(self, msg):
        self.gripper_state_deque.append(msg)
        # self.gripper_position = msg.gPO  # Current position
        # self.gripper_status = msg.gSTA   # Gripper status
        # self.object_detected = msg.gOBJ  # Object detection status


    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False

        # Get latest frames
        img_left = imgmsg_to_cv2(self.img_left_deque[-1])
        img_right = imgmsg_to_cv2(self.img_right_deque[-1])
        img_front = imgmsg_to_cv2(self.img_front_deque[-1])
        puppet_arm_right = self.puppet_arm_right_deque[-1]
        gripper_state = self.gripper_state_deque[-1]

        return (img_front, img_left, img_right,
                puppet_arm_right, gripper_state)


def model_inference(args, config, ros_operator):
    global lang_embeddings

    policy = make_policy(args)
    lang_dict = torch.load(args.lang_embeddings_path)
    # print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    # lang_embeddings = lang_dict["embeddings"]
    lang_embeddings = lang_dict["Pick up the stress ball."]

    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initialize position of the puppet arm
    right0 = [-0.16890989371977355, -0.6588316862875955, 0.10070957168356454, -2.6189301301038976, -0.00583621904111755, 1.97059170823627, 1.4790283997696552]

    right1 = [-0.16890989371977355, -0.6588316862875955, 0.10070957168356454, -2.6189301301038976, -0.00583621904111755, 1.97059170823627, 1.4790283997696552]
    
    ros_operator.init_gripper()

    ros_operator.puppet_arm_publish_continuous(right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(right1)

    pre_action = np.zeros(config['state_dim'])
    pre_action[:7] = np.array(right1)

    input("Press enter to start inference")
    with torch.inference_mode():
        while not rospy.is_shutdown():
            print("inference mode")
            t = 0
            rate = rospy.Rate(args.publish_rate)
            action_buffer = np.zeros([chunk_size, config['state_dim']])

            while t < max_publish_step and not rospy.is_shutdown():
                update_observation_window(args, config, ros_operator)

                if t % chunk_size == 0:
                    action_buffer = inference_fn(args, config, policy, t).copy()
                    print("start inference")
                
                # if t % chunk_size > 32:
                #     t+=1
                #     print("skip execution")
                #     continue

                raw_action = action_buffer[t % chunk_size]
                action = raw_action

                if args.use_actions_interpolation:
                    interp_actions = interpolate_action(args, pre_action, action)
                else:
                    interp_actions = action[np.newaxis, :]

                print(f"interp_actions:{interp_actions}")
                for act in interp_actions:
                    right_action = act[:8]

                    if not args.disable_puppet_arm:
                        ros_operator.puppet_arm_publish(right_action)

                    rate.sleep()

                t += 1
                print("Published Step", t)
                pre_action = action.copy()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int,
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera2/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera1/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera1/color/image_raw', required=False)

    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/init_pose', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/joint_states', required=False)
    parser.add_argument('--gripper_state_topic', action='store', type=str, help='gripper_state_topic',
                        default='/Robotiq2FGripperRobotInput', required=False)
    parser.add_argument('--gripper_cmd_topic', action='store', type=str, help='gripper_cmd_topic',
                        default='/Robotiq2FGripperRobotOutput', required=False)

    parser.add_argument('--publish_rate', action='store', type=int,
                        help='The rate at which to publish the actions',
                        default=3, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int,
                        help='The control frequency of the robot',
                        default=10, required=False)

    parser.add_argument('--chunk_size', action='store', type=int,
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Whether to use depth images',
                        default=False, required=False)

    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging', default=False)

    parser.add_argument('--config_path', type=str, default="../configs/base.yaml",
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=False, default = "../checkpoints/checkpoint-120000",
                        help='Name or path to the pretrained model')

    parser.add_argument('--lang_embeddings_path', type=str, required=False, default= "../outs/instruction_embeddings.pt",
                        help='Path to the pre-encoded language instruction embeddings')

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    rospy.init_node('joint_state_publisher', anonymous=True)
    ros_operator = RosOperator(args)

    if args.seed is not None:
        set_seed(args.seed)

    config = get_config(args)

    try:
        inference_thread = threading.Thread(target=model_inference, args=(args, config, ros_operator))
        inference_thread.start()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        inference_thread.join()


if __name__ == '__main__':
    main()

    # args = get_arguments()
    # lang_dict = torch.load(args.lang_embeddings_path)
    # print(lang_dict.keys())


    # ckpt_path = "../checkpoints/checkpoint-1600"
    # lang_embedding_path = "../outs/insturction_embeddings.pt"

    # args = get_arguments()

    # # args.pretrained_model_name_or_path = ckpt_path
    # # args.lang_embeddings_path = lang_embedding_path

    # rospy.init_node('joint_state_publisher', anonymous=True)
    # ros_operator = RosOperator(args)
    # ros_operator.init_ros()
    # result = ros_operator.get_frame()
    # print(result)

    # try:
    #     rospy.spin()
        
    # except KeyboardInterrupt:
    #     print("Shutting down")
        