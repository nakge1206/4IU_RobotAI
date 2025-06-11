#!/usr/bin/env python3

import rospy
import tkinter as tk
import threading
from std_msgs.msg import Int16MultiArray

class MotorControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOMI Motor Control")

        rospy.init_node('gui_publisher', anonymous=True)

        # 퍼블리셔 생성 (모터 속도 변경)
        self.motor_speed_pub = rospy.Publisher('/motor_speed', Int16MultiArray, queue_size=10)

        # 현재 모터 속도를 가져오는 요청 퍼블리셔
        self.motor_speed_request_pub = rospy.Publisher('/motor_speed_request', Int16MultiArray, queue_size=10)

        # 모터 속도 데이터 저장
        self.motor_speeds = [0] * 10  
        self.initial_length = 10  # 기본적으로 10개의 모터가 있다고 가정

        # GUI 요소 생성
        tk.Label(root, text="Motor Speed Control", font=("Arial", 14)).grid(row=0, columnspan=4)

        # "Load Motor Speeds" 버튼 (현재 값 불러오기)
        self.load_button = tk.Button(root, text="Load Motor Speeds", command=self.request_motor_speeds)
        self.load_button.grid(row=1, column=0, columnspan=4, pady=5)

        # 슬라이더 저장 리스트
        self.sliders = []

        # 슬라이더 UI 생성
        self.create_sliders()

        # "적용" 버튼 (수정된 값 반영)
        self.apply_button = tk.Button(root, text="Apply Changes", command=self.apply_motor_speeds)
        self.apply_button.grid(row=self.initial_length + 2, column=0, columnspan=4, pady=10)

        # ROS 구독 설정 (모터 속도 업데이트)
        rospy.Subscriber('/motor_speed', Int16MultiArray, self.update_motor_speeds)

    def create_sliders(self):
        """모터 속도를 조절할 슬라이더 UI + 버튼 생성"""
        for i in range(self.initial_length):
            tk.Label(self.root, text=f"Motor {i}").grid(row=i+2, column=0)

            # 감소 버튼 (-)
            minus_button = tk.Button(self.root, text="-", command=lambda idx=i: self.adjust_slider(idx, -1))
            minus_button.grid(row=i+2, column=1)

            # 슬라이더
            slider = tk.Scale(self.root, from_=0, to=255, orient="horizontal")
            slider.grid(row=i+2, column=2)
            self.sliders.append(slider)  # 리스트에 슬라이더 추가

            # 증가 버튼 (+)
            plus_button = tk.Button(self.root, text="+", command=lambda idx=i: self.adjust_slider(idx, 1))
            plus_button.grid(row=i+2, column=3)

    def adjust_slider(self, index, change):
        """+ 또는 - 버튼을 눌렀을 때 슬라이더 값을 조정"""
        current_value = self.sliders[index].get()
        new_value = max(0, min(255, current_value + change))  # 0~255 범위 제한
        self.sliders[index].set(new_value)

    def update_motor_speeds(self, msg):
        """ROS에서 받은 모터 속도 데이터를 슬라이더에 반영"""
        new_data = list(msg.data)

        # 받은 데이터가 모터 개수와 다르면 무시
        if len(new_data) != self.initial_length:
            return

        # 불러온 값을 슬라이더에 적용
        for i in range(self.initial_length):
            self.motor_speeds[i] = new_data[i]  # 리스트 업데이트
            self.sliders[i].set(new_data[i])  # 슬라이더 UI 업데이트

        print("Motor Speeds Updated:", self.motor_speeds)

    def request_motor_speeds(self):
        """아두이노에 현재 모터 속도 요청"""
        msg = Int16MultiArray()
        msg.data = []  # 데이터 없이 요청만 보냄
        self.motor_speed_request_pub.publish(msg)
        print("Requested current motor speeds from Arduino.")

    def apply_motor_speeds(self):
        """슬라이더 값을 읽어 모터 속도를 업데이트하고 ROS 토픽 발행"""
        for i in range(self.initial_length):
            self.motor_speeds[i] = self.sliders[i].get()  # 슬라이더에서 값 가져오기

        msg = Int16MultiArray()
        msg.data = self.motor_speeds[:]  # 리스트 데이터 복사하여 발행
        self.motor_speed_pub.publish(msg)

        print("Applied Motor Speeds:", self.motor_speeds)

# ROS가 실행 중인지 확인
if __name__ == "__main__":
    root = tk.Tk()
    app = MotorControlGUI(root)

    # ROS 메시지를 백그라운드에서 받아들이도록 설정
    def ros_spin():
        rate = rospy.Rate(10)  # 10Hz (0.1초마다 메시지 확인)
        while not rospy.is_shutdown():
            rate.sleep()
            rospy.rostime.wallsleep(0.1)

    thread = threading.Thread(target=ros_spin)
    thread.daemon = True
    thread.start()

    root.mainloop()
