#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import os, glob, time
from std_msgs.msg import Int16MultiArray, Bool

#스피커 사용 -> 스레드로 순차 처리해야함
import queue
import threading
import pyttsx3

class MotionSequenceExecutor:
    def __init__(self):
        rospy.init_node('main_core', anonymous=True)
        rospy.Subscriber('/play_motion_sequence', String, self.handle_sequence_request)

        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 160)

        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.tts_loop, daemon=True)
        self.tts_thread.start()


        # 4개의 스위치 상태 구독 (추후 기능 개발 예정)
        self.switch_subs = []
        for i in range(4):
            topic = f"/switch_{i+1}_state"
            sub = rospy.Subscriber(topic, Bool, self.switch_callback, callback_args=i)
            self.switch_subs.append(sub)

        rospy.loginfo("✅ Main Core Node Initialized")
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
    

    def tts_loop(self):
        while not rospy.is_shutdown():
            text = self.tts_queue.get()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.tts_queue.task_done()


    def switch_callback(self, msg, index):
        state = "ON" if msg.data else "OFF"
        rospy.loginfo(f"📶 Switch {index + 1} is {state}")
    
        if msg.data:  # ON일 때만 말하게
            message_map = {
                0: "back sensor", # sw1
                1: "left arm",    # sw2
                2: "right arm",   # sw3 → 나중에 추가 가능
                3: "head"         # sw4 → 나중에 추가 가능
            }
            if index in message_map:
                self.tts_queue.put(message_map[index])
    
if __name__ == '__main__':
    try:
        MotionSequenceExecutor()
    except rospy.ROSInterruptException:
        pass
