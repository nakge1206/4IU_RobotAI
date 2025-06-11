#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy #joystick 값
from std_msgs.msg import String, Int16MultiArray, Bool

import json
import os
import time

# 자율주행파트
# from geometry_msgs.msg import PoseStamped
# import tf.transformations
# import glob
# import math


class MotionSequenceExecutor:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('yomi_motor', anonymous=True)

        #Subscriber
        rospy.Subscriber('/play_motion_sequence', String, self.handle_sequence_request)
        rospy.Subscriber('/switch_1_state', Bool, self.switch_callback, callback_args=1)
        rospy.Subscriber('/switch_2_state', Bool, self.switch_callback, callback_args=2)
        rospy.Subscriber('/switch_3_state', Bool, self.switch_callback, callback_args=3)
        rospy.Subscriber('/switch_4_state', Bool, self.switch_callback, callback_args=4)
        rospy.Subscriber('/switch_5_state', Bool, self.switch_callback, callback_args=5)
        rospy.Subscriber('/switch_6_state', Bool, self.switch_callback, callback_args=6)
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        #영상처리
        # rospy.Subscriber('/detected_object', String, self.handle_detected_object)

        # Publisher
        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)
        
        #자율주행
        # self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        #Joystick
        self.prev_buttons = [0] * 10 #Joystick의 버튼의 상태변화를 저장하기 위한 변수
        

        # # 객체 인식 -> 목표 위치 -> 특정 시퀀스 실행
        # self.target_map = {
        #     "person": ((1.0, 2.0, 90), "/home/micca/catkin_ws/src/yomi/motion/hi")
        #     #"bottle": ((2.5, 0.5, 180), "/home/micca/catkin_ws/src/yomi/motion/hi"),
        #     #"cup": ((0.5, -1.0, 0), "/home/micca/catkin_ws/src/yomi/motion/hi")
        # }
        # self.sent_labels = set()

        rospy.loginfo("✅ Main Core Node with Object Navigation Initialized")
        rospy.spin()

    def joy_callback(self, msg):
        """
        f710버튼 pressed, released를 구분하는 joystick콜백함수
        stick값 확장가능
        """
        for i in [3, 1]:  # Y (3), B (1)
            if msg.buttons[i] == 1 and self.prev_buttons[i] == 0:
                if i == 3:
                    rospy.loginfo("Y pressed")
                elif i == 1:
                    rospy.loginfo("B pressed")
            if msg.buttons[i] == 0 and self.prev_buttons[i] == 1:
                if i == 3:
                    rospy.loginfo("Y released")
                elif i == 1:
                    rospy.loginfo("B released")
        self.prev_buttons = list(msg.buttons)

    def handle_sequence_request(self, msg):
        """동작 json파일을 읽어서 speed, position, servo, time값을 excute_sequence로 전송"""
        emotion = msg.data.strip()
        base_path = "/home/micca/catkin_ws/src/yomi/motion"
        file_path = os.path.join(base_path, f"{emotion}.json")

        if not os.path.isfile(file_path):
            rospy.logwarn(f"⚠️ Invalid folder path: {file_path}")
            return

        rospy.loginfo(f"▶ Executing motion sequence for emotion: {emotion}")
        self.execute_sequence(file_path)

    def execute_sequence(self, json_path):
        """
        모션.json파일 내에 있는 값들을 읽고, topic에 전달
        모터속도(10개) : 0~9
        모터위치(10개) : 10~19
        서보각도(8개) : 20~27
        타임스탬프(1개) : 28 <- 타임스탬프는 각 모터의 저장을 각 시간별로 저장한 것
        """
        try:
            with open(json_path, "r") as f:
                motion_list = json.load(f)
        except Exception as e:
            rospy.logerr(f"Failed to load JSON: {e}")
            return
        
        start_time = time.time()

        for motion in motion_list:
            try:
                timestamp_sec = motion["timestamp_ms"] / 1000.0

                # 대기: 타이밍 맞추기
                while time.time() - start_time < timestamp_sec:
                    time.sleep(0.001)

                # 각 모터 ID 정렬 후 순차 적용
                motor_ids = sorted([int(k) for k in motion["motor_speeds"].keys()])
                motor_speeds = [motion["motor_speeds"][str(k)] for k in motor_ids]
                motor_positions = [motion["motor_positions"][str(k)] for k in motor_ids]

                servo_ids = sorted([int(k) for k in motion["servo_angles"].keys()])
                servo_angles = [motion["servo_angles"][str(k)] for k in servo_ids]

                # ROS 토픽 퍼블리시
                self.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
                self.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
                self.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))

                rospy.loginfo(f"✅ Executed motion at {motion['timestamp_ms']} ms")

            except Exception as e:
                rospy.logerr(f"❌ Error executing motion block: {e}")

    def switch_callback(self, msg, index):
        """
        yomi몸에 달려있는 switch눌렸을때 콜백함수
        0: 등
        1: 왼팔
        2: 오른팔
        3: 왼손
        4: 오른손
        5: 머리
        """
        state = True if msg.data else False
        rospy.loginfo(f"Switch {index} if {str(state)}")


    #영상처리
    # def handle_detected_object(self, msg):
    #     """yolo v5의 detect.py를 실행하면 내부에 rosnode 자동생성되어 /detected_object topic으로 퍼블리시해줌"""
    #     label = msg.data.strip()
    #     if label in self.target_map and label not in self.sent_labels:
    #         (x, y, yaw_deg), sequence_path = self.target_map[label]

    #         self.tts_queue.put(f"{label} detected")
            
    #         self.send_goal(x, y, yaw_deg)
    #         self.sent_labels.add(label)

    #         # 시퀀스 실행을 위치 도달 후로 지연 (현재는 Duration 으로 지연함. 개발 필요시 도착했는지 토픽 받아서 제어하면 됨)
    #         rospy.Timer(rospy.Duration(8), lambda event: self.execute_sequence_from_path(sequence_path), oneshot=True)



    # def send_goal(self, x, y, yaw_deg):
    #     """자율주행 파트"""
    #     goal = PoseStamped()
    #     goal.header.stamp = rospy.Time.now()
    #     goal.header.frame_id = "map"

    #     goal.pose.position.x = x
    #     goal.pose.position.y = y
    #     goal.pose.position.z = 0.0

    #     yaw_rad = math.radians(yaw_deg)
    #     quat = tf.transformations.quaternion_from_euler(0, 0, yaw_rad)
    #     goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = quat

    #     self.goal_pub.publish(goal)
    #     rospy.loginfo(f"📍 Sent goal to ({x:.2f}, {y:.2f}, {yaw_deg:.1f}°)")
    #     self.tts_queue.put(f"Moving to {x:.1f}, {y:.1f}, heading {int(yaw_deg)} degrees")

if __name__ == '__main__':
    try:
        MotionSequenceExecutor()
    except rospy.ROSInterruptException:
        pass
