#!/usr/bin/env python3

import rospy
import os, glob, time, math, queue, threading
from std_msgs.msg import String, Int16MultiArray, Bool
from geometry_msgs.msg import PoseStamped
import tf.transformations
#import pyttsx3
import asyncio
import edge_tts
import playsound


class MotionSequenceExecutor:
    def __init__(self):
        rospy.init_node('yomi_main', anonymous=True)

        # Subscriber
        rospy.Subscriber('/play_motion_sequence', String, self.handle_sequence_request)
        rospy.Subscriber('/detected_object', String, self.handle_detected_object)

        # Publisher
        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # TTS->스레드로 처리
        #self.tts_engine = pyttsx3.init()
        #self.tts_engine.setProperty('rate', 160)
        self.tts_queue = queue.Queue()
        threading.Thread(target=self.tts_loop, daemon=True).start()

        # 스위치 상태 Subscriber
        self.switch_subs = []
        for i in range(6):
            topic = f"/switch_{i+1}_state"
            sub = rospy.Subscriber(topic, Bool, self.switch_callback, callback_args=i)
            self.switch_subs.append(sub)

        # 객체 인식 -> 목표 위치 -> 특정 시퀀스 실행
        self.target_map = {
            "person": ((1.0, 2.0, 90), "/home/micca/catkin_ws/src/yomi/motion/hi")
            #"bottle": ((2.5, 0.5, 180), "/home/micca/catkin_ws/src/yomi/motion/hi"),
            #"cup": ((0.5, -1.0, 0), "/home/micca/catkin_ws/src/yomi/motion/hi")
        }
        self.sent_labels = set()

        rospy.loginfo("✅ Main Core Node with Object Navigation Initialized")
        rospy.spin()

    def handle_sequence_request(self, msg):
        folder_path = msg.data.strip()
        if not os.path.isdir(folder_path):
            rospy.logwarn(f"⚠️ Invalid folder path: {folder_path}")
            return

        file_list = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
        if not file_list:
            rospy.logwarn(f"⚠️ No motion files found in: {folder_path}")
            return

        rospy.loginfo(f"▶ Executing {len(file_list)} motions from: {folder_path}")
        self.execute_sequence(file_list)

    def execute_sequence(self, file_list):
        start_time = time.time()
        for filepath in file_list:
            try:
                with open(filepath, "r") as f:  
                    values = [float(x) for x in f.readline().strip().split(",")]
                if len(values) != 29:
                    raise ValueError("Expected 29 values (10 speed + 10 pos + 8 angle + 1 timestamp)")

                timestamp_sec = values[28] / 1000.0
                motor_speeds = list(map(int, values[0:10]))
                motor_positions = list(map(int, values[10:20]))
                servo_angles = list(map(int, values[20:28]))

                while time.time() - start_time < timestamp_sec:
                    time.sleep(0.001)

                self.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
                self.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
                self.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))
                rospy.loginfo(f"✅ Executed {os.path.basename(filepath)}")

            except Exception as e:
                rospy.logerr(f"❌ Failed to execute {filepath}: {e}")

    def handle_detected_object(self, msg):
        label = msg.data.strip()
        if label in self.target_map and label not in self.sent_labels:
            (x, y, yaw_deg), sequence_path = self.target_map[label]

            self.tts_queue.put(f"{label} detected")
            
            self.send_goal(x, y, yaw_deg)
            self.sent_labels.add(label)

            # 시퀀스 실행을 위치 도달 후로 지연 (현재는 Duration 으로 지연함. 개발 필요시 도착했는지 토픽 받아서 제어하면 됨)
            rospy.Timer(rospy.Duration(8), lambda event: self.execute_sequence_from_path(sequence_path), oneshot=True)


    def execute_sequence_from_path(self, folder_path):
        if not os.path.isdir(folder_path):
            rospy.logwarn(f"⚠️ Invalid sequence path: {folder_path}")
            return
    
        file_list = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
        if not file_list:
            rospy.logwarn(f"⚠️ No motion files in: {folder_path}")
            return
    
        rospy.loginfo(f"▶ Executing sequence from {folder_path}")
        self.tts_queue.put(f"Executing sequence")
        self.execute_sequence(file_list)

    def send_goal(self, x, y, yaw_deg):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        yaw_rad = math.radians(yaw_deg)
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw_rad)
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = quat

        self.goal_pub.publish(goal)
        rospy.loginfo(f"📍 Sent goal to ({x:.2f}, {y:.2f}, {yaw_deg:.1f}°)")
        self.tts_queue.put(f"Moving to {x:.1f}, {y:.1f}, heading {int(yaw_deg)} degrees")

    def tts_loop(self):
        while not rospy.is_shutdown():
            text = self.tts_queue.get()
            try:
                filename = "/tmp/edge_tts_output.mp3"
                asyncio.run(self.generate_tts(text, filename))
                playsound.playsound(filename)
            except Exception as e:
                rospy.logerr(f"❌ TTS failed: {e}")
            self.tts_queue.task_done()
        """
        while not rospy.is_shutdown():
            text = self.tts_queue.get()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.tts_queue.task_done()
            """
    async def generate_tts(self, text, filename):
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")  # ▶ 한국어면 "ko-KR-SunHiNeural"
        await communicate.save(filename)

    def switch_callback(self, msg, index):
        state = "ON" if msg.data else "OFF"
        rospy.loginfo(f"📶 Switch {index + 1} is {state}")
        if msg.data:
            message_map = {
                0: "back sensor",
                1: "left arm",
                2: "right arm",
                3: "left hand",
                4: "right hand",
                5: "head"
            }
            if index in message_map:
                self.tts_queue.put(message_map[index])

if __name__ == '__main__':
    try:
        MotionSequenceExecutor()
    except rospy.ROSInterruptException:
        pass
