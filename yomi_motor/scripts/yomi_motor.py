#!/usr/bin/env python3

import rospy
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

        # Publisher
        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)

        self.motor_speed_request_pub = rospy.Publisher('/motor_speed_request', Int16MultiArray, queue_size=10)
        self.motor_position_request_pub = rospy.Publisher('/motor_position_request', Int16MultiArray, queue_size=10)
        self.servo_angle_request_pub = rospy.Publisher('/servo_angle_request', Int16MultiArray, queue_size=8)


        #Subscriber
        # rospy.Subscriber('/play_motion_sequence', String, self.handle_sequence_request)
        # rospy.Subscriber('/switch_1_state', Bool, self.switch_callback, callback_args=1)
        # rospy.Subscriber('/switch_2_state', Bool, self.switch_callback, callback_args=2)
        # rospy.Subscriber('/switch_3_state', Bool, self.switch_callback, callback_args=3)
        # rospy.Subscriber('/switch_4_state', Bool, self.switch_callback, callback_args=4)
        # rospy.Subscriber('/switch_5_state', Bool, self.switch_callback, callback_args=5)
        # rospy.Subscriber('/switch_6_state', Bool, self.switch_callback, callback_args=6)
        self.motor_id = 3
        self.motor_speed = 2
        self.motor_pos = 180
        
        self.motor_limits = {
            3: (120, 300), 4: (110, 260), 5: (90, 270), 6: (95, 180),
            7: (60, 240), 8: (100, 250), 9: (90, 270),
            10: (180, 265), 11: (160, 240), 12: (110, 250)
        }

        #현재 모터 정보
        self.curMotorSpeed = [2] * 10 
        self.curMotorPos = [180, 110, 180, 180, 180, 250, 180, 180, 190, 180]
        self.curServoAngle = [90] * 8

        rospy.Subscriber('/motor_speed_fb', Int16MultiArray, self.motor_callback, callback_args=1)
        rospy.Subscriber('/motor_position_fb', Int16MultiArray, self.motor_callback, callback_args=2)
        rospy.Subscriber('/servo_angle_fb', Int16MultiArray, self.motor_callback, callback_args=3)
        

        #영상처리
        # rospy.Subscriber('/detected_object', String, self.handle_detected_object)


        #모션입력 5초동안 안들어오면 차렷자세
        # self.last_received_time = time.time()  # 마지막 메시지 수신 시각 초기화
        # self.idle_timer = rospy.Timer(rospy.Duration(1), self.check_idle)  # 1초마다 체크
        # self.idle_pub = rospy.Publisher('/play_motion_sequence', String, queue_size=1)  # 원하는 토픽 이름
        # self.max_motion_delay = 11


        
        #자율주행
        # self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        

        # # 객체 인식 -> 목표 위치 -> 특정 시퀀스 실행
        # self.target_map = {
        #     "person": ((1.0, 2.0, 90), "/home/micca/catkin_ws/src/yomi/motion/hi")
        #     #"bottle": ((2.5, 0.5, 180), "/home/micca/catkin_ws/src/yomi/motion/hi"),
        #     #"cup": ((0.5, -1.0, 0), "/home/micca/catkin_ws/src/yomi/motion/hi")
        # }
        # self.sent_labels = set()

        rospy.loginfo("✅ Main Core Node with Object Navigation Initialized")
        # rospy.spin()

    # def check_idle(self, event):
    #     if time.time() - self.last_received_time > self.max_motion_delay:
    #         rospy.logwarn(f"play_motion_sequence 토픽이 {self.max_motion_delay}초간 비활성 상태입니다.")
    #         self.idle_pub.publish("stand_up")
    #         self.last_received_time = time.time()  # 중복 퍼블리시 방지 (한 번만 보내고 다시 5초 기다림)

    # def handle_sequence_request(self, msg):
    #     """동작 json파일을 읽어서 speed, position, servo, time값을 excute_sequence로 전송"""
    #     emotion = msg.data.strip()
    #     base_path = "/home/micca/catkin_ws/src/4IU_RobotAI/yomi_motor/motion"
    #     file_path = os.path.join(base_path, f"{emotion}.json")

    #     if not os.path.isfile(file_path):
    #         rospy.logwarn(f"⚠️ Invalid folder path: {file_path}")
    #         return

    #     rospy.loginfo(f"▶ Executing motion sequence for emotion: {emotion}")
    #     self.execute_sequence(file_path)

    # def execute_sequence(self, json_path):
    #     """
    #     모션.json파일 내에 있는 값들을 읽고, topic에 전달
    #     모터속도(10개) : 0~9
    #     모터위치(10개) : 10~19
    #     서보각도(8개) : 20~27
    #     타임스탬프(1개) : 28 <- 타임스탬프는 각 모터의 저장을 각 시간별로 저장한 것
    #     """
    #     try:
    #         with open(json_path, "r") as f:
    #             motion_list = json.load(f)
    #     except Exception as e:
    #         rospy.logerr(f"Failed to load JSON: {e}")
    #         return
        
    #     start_time = time.time()

    #     for motion in motion_list:
    #         try:
    #             timestamp_sec = motion["timestamp_ms"] / 1000.0

    #             # 대기: 타이밍 맞추기
    #             while time.time() - start_time < timestamp_sec:
    #                 time.sleep(0.001)

    #             # 각 모터 ID 정렬 후 순차 적용
    #             motor_ids = sorted([int(k) for k in motion["motor_speeds"].keys()])
    #             motor_speeds = [motion["motor_speeds"][str(k)] for k in motor_ids]
    #             motor_positions = [motion["motor_positions"][str(k)] for k in motor_ids]

    #             servo_ids = sorted([int(k) for k in motion["servo_angles"].keys()])
    #             servo_angles = [motion["servo_angles"][str(k)] for k in servo_ids]

    #             # ROS 토픽 퍼블리시
    #             self.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
    #             self.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
    #             self.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))

    #             rospy.loginfo(f"✅ Executed motion at {motion['timestamp_ms']} ms")
    #             self.last_received_time = time.time()

    #         except Exception as e:
    #             rospy.logerr(f"❌ Error executing motion block: {e}")

    # def switch_callback(self, msg, index):
    #     """
    #     yomi몸에 달려있는 switch눌렸을때 콜백함수
    #     1: 등
    #     2: 왼팔
    #     3: 오른팔
    #     4: 왼손
    #     5: 오른손
    #     6: 머리

    #     1: 등 - 정보출력
    #     2: 왼팔 - pos+1
    #     3: 오른팔 - pos-1
    #     4: 왼손 - pos+5
    #     5: 오른손 - pos-5
    #     6: 머리 - motor변경
    #     """
    #     if not msg.data:
    #         return  # 버튼 떼면 무시
    #     state = True if msg.data else False
    #     if state:
    #         rospy.loginfo(f"Switch {index} if {str(state)}")
        
    #     if index == 1:
    #         self.request_all_data()

        # elif index == 6:
        #     # 모터 변경
        #     self.motor_id = 3 if self.motor_id == 12 else self.motor_id + 1
        #     rospy.loginfo(f"motor_id = {self.motor_id}")

        #     default_positions = {
        #         3: 180, 7: 120, 
        #         4: 180, 8: 180,
        #         5: 180, 9: 240, 
        #         6: 180, 10: 180, 
        #         11: 190, 
        #         12: 180
        #     }
        #     self.motor_pos = default_positions.get(self.motor_id)
        #     rospy.loginfo(f"[기본 위치 설정] motor_id={self.motor_id}, motor_pos={self.motor_pos}")

        # elif index in [2, 3, 4, 5]:
        #     # 증감량 정의
        #     delta = {2: 1, 3: -1, 4: 5, 5: -5}.get(index, 0)
        #     self.motor_pos += delta

        #     # 현재 모터의 범위 제한 적용
        #     min_pos, max_pos = self.motor_limits.get(self.motor_id, (0, 300))
        #     if self.motor_pos < min_pos:
        #         self.motor_pos = min_pos
        #         rospy.logwarn(f"⬅️ 모터 {self.motor_id} 최소값 {min_pos} 도달")
        #     elif self.motor_pos > max_pos:
        #         self.motor_pos = max_pos
        #         rospy.logwarn(f"➡️ 모터 {self.motor_id} 최대값 {max_pos} 도달")

        #     # 퍼블리시
        #     self.motor_publisher(
        #         motor_id=self.motor_id,
        #         motor_position=self.motor_pos,
        #         motor_speed=self.motor_speed
        #     )


    
    # def motor_publisher(self, motor_id = None, motor_speed = None, motor_position = None, servo_id = None, servo_angle = None):
    #     """
    #     특정 모터 또는 서보의 속도/위치/각도를 퍼블리시합니다.
    #     - 전달된 값(motor_speed, motor_position, servo_angle)이 있으면 그것을 사용
    #     - 없으면 내부 상태(self.curXXX)에서 값을 가져와 퍼블리시

    #     Args:
    #         motor_id: 3~12 (10개의 DC모터)
    #         servo_id: 1,2,3,4,6,7,8,9 (8개의 서보)
    #     """
    #     # self.request_all_data()
    #     if motor_id is not None:
    #         if 3 <= motor_id <= 12:
    #             index = motor_id - 3
    #             speed = motor_speed if motor_speed is not None else self.curMotorSpeed[index]
    #             pos = motor_position if motor_position is not None else self.curMotorPos[index]

    #             speed_msg = Int16MultiArray(data=self.curMotorSpeed.copy())
    #             pos_msg = Int16MultiArray(data=self.curMotorPos.copy())
    #             speed_msg.data[index] = speed
    #             pos_msg.data[index] = pos

    #             self.motor_speed_pub.publish(speed_msg)
    #             self.motor_position_pub.publish(pos_msg)

    #     if servo_id is not None:
    #         servo_ids = [1, 2, 3, 4, 6, 7, 8, 9]
    #         try:
    #             index = servo_ids.index(servo_id)
    #             angle = servo_angle if servo_angle is not None else self.curServoAngle[index]

    #             angle_msg = Int16MultiArray(data=self.curServoAngle.copy())
    #             angle_msg.data[index] = angle

    #             self.servo_angle_pub.publish(angle_msg)
    #             rospy.loginfo(f"📤 Published servo ID {servo_id} | angle={angle}")

    #         except ValueError:
    #             rospy.logwarn(f"❌ Invalid servo_id: {servo_id}")

    def request_all_data(self):
        self.motor_speed_request_pub.publish(Int16MultiArray(data=[]))
        self.motor_position_request_pub.publish(Int16MultiArray(data=[]))
        self.servo_angle_request_pub.publish(Int16MultiArray(data=[]))

    def motor_callback(self, msg, index):
        """
        index : 
            1 = MoterSpeed
            2 = MotorPosition
            3 = ServoAngle
        """
        if index == 1:  # 모터 속도
            self.curMotorSpeed = list(msg.data)
            # rospy.loginfo(f"[motor_callback] Updated curMotorSpeed: {list(msg.data)}")
        elif index == 2:  # 모터 위치
            self.curMotorPos = list(msg.data)
            rospy.loginfo(f"[motor_callback] Updated curMotorPos: {list(msg.data)}")
        elif index == 3:  # 서보 각도
            self.curServoAngle = list(msg.data)
            # rospy.loginfo(f"[motor_callback] Updated curServoAngle: {list(msg.data)}")
            
    def motor_publisher_batch(self, motor_ids=[], motor_speeds=[], motor_positions=[], servo_ids=[], servo_angles=[]):
        """
        여러 모터 또는 서보의 속도/위치/각도를 한 번에 퍼블리시합니다.
        """
        # 모터
        if motor_ids:
            speed_data = self.curMotorSpeed.copy()
            pos_data = self.curMotorPos.copy()

            for i, motor_id in enumerate(motor_ids):
                index = motor_id - 3
                if index < 0 or index >= len(speed_data):
                    rospy.logwarn(f"❌ 잘못된 motor_id: {motor_id}")
                    continue

                if i < len(motor_speeds):
                    speed_data[index] = motor_speeds[i]
                if i < len(motor_positions):
                    pos_data[index] = motor_positions[i]

            self.motor_speed_pub.publish(Int16MultiArray(data=speed_data))
            self.motor_position_pub.publish(Int16MultiArray(data=pos_data))

        # 서보
        if servo_ids:
            angle_data = self.curServoAngle.copy()
            valid_ids = [1, 2, 3, 4, 6, 7, 8, 9]

            for i, sid in enumerate(servo_ids):
                if sid not in valid_ids:
                    rospy.logwarn(f"❌ 잘못된 servo_id: {sid}")
                    continue

                index = valid_ids.index(sid)
                if i < len(servo_angles):
                    angle_data[index] = servo_angles[i]

            self.servo_angle_pub.publish(Int16MultiArray(data=angle_data))


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

def main():
    node = MotionSequenceExecutor()
    # rospy.loginfo("✅ Main Core Node with Object Navigation Initialized")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass