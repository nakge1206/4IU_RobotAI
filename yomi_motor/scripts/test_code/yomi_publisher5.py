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

        # 퍼블리셔 (명령 전송)
        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)

        # 현재 상태 요청 퍼블리셔
        self.motor_speed_request_pub = rospy.Publisher('/motor_speed_request', Int16MultiArray, queue_size=10)
        self.motor_position_request_pub = rospy.Publisher('/motor_position_request', Int16MultiArray, queue_size=10)
        self.servo_angle_request_pub = rospy.Publisher('/servo_angle_request', Int16MultiArray, queue_size=10)

        # 데이터 저장 및 개수 설정
        self.motor_count = 10
        self.servo_count = 8
        self.motor_speeds = [0] * self.motor_count  
        self.motor_positions = [0] * self.motor_count  
        self.servo_angles = [0] * self.servo_count  

        # 상단 타이틀 및 "Load Data" 버튼
        tk.Label(root, text="Motor Speed Control", font=("Arial", 14)).grid(row=0, columnspan=4)
        self.load_button = tk.Button(root, text="Load Data", command=self.request_all_data)
        self.load_button.grid(row=1, column=0, columnspan=4, pady=5)

        # 모터 슬라이더 (속도, 위치)
        self.speed_sliders = []
        self.position_sliders = []
        self.create_motor_sliders()

        # 모터 적용 버튼
        self.apply_motor_button = tk.Button(root, text="Apply Motor Speed & Position", command=self.apply_motor_data)
        self.apply_motor_button.grid(row=self.motor_count+2, column=0, columnspan=4, pady=5)

        # 서보 슬라이더 섹션
        start_row = self.motor_count + 4
        tk.Label(root, text="Servo Angle Control", font=("Arial", 14)).grid(row=start_row, columnspan=2, pady=(10, 0))

        # 서보 슬라이더
        self.angle_sliders = []
        self.create_servo_sliders(start_row + 1)

        # 서보 적용 버튼
        self.apply_servo_button = tk.Button(root, text="Apply Servo Angles", command=self.apply_servo_data)
        self.apply_servo_button.grid(row=start_row + self.servo_count + 1, column=0, columnspan=2, pady=5)

        # ROS 구독 (피드백)
        rospy.Subscriber('/motor_speed_fb', Int16MultiArray, self.update_motor_speeds)
        rospy.Subscriber('/motor_position_fb', Int16MultiArray, self.update_motor_positions)
        rospy.Subscriber('/servo_angle_fb', Int16MultiArray, self.update_servo_angles)

    def create_motor_sliders(self):
        for i in range(self.motor_count):
            tk.Label(self.root, text=f"Motor {i}").grid(row=i+2, column=0)
            speed_slider = tk.Scale(self.root, from_=0, to=255, orient="horizontal")
            speed_slider.grid(row=i+2, column=1)
            self.speed_sliders.append(speed_slider)

            position_slider = tk.Scale(self.root, from_=-100, to=100, orient="horizontal")
            position_slider.grid(row=i+2, column=2)
            self.position_sliders.append(position_slider)

    def create_servo_sliders(self, start_row):
        for i in range(self.servo_count):
            tk.Label(self.root, text=f"Servo {i}").grid(row=start_row+i, column=0)
            angle_slider = tk.Scale(self.root, from_=0, to=180, orient="horizontal")
            angle_slider.grid(row=start_row+i, column=1)
            self.angle_sliders.append(angle_slider)

    def update_motor_speeds(self, msg):
        for i, value in enumerate(msg.data):
            if i < self.motor_count:
                self.motor_speeds[i] = value
                self.speed_sliders[i].set(value)

    def update_motor_positions(self, msg):
        for i, value in enumerate(msg.data):
            if i < self.motor_count:
                self.motor_positions[i] = value
                self.position_sliders[i].set(value)

    def update_servo_angles(self, msg):
        for i, value in enumerate(msg.data):
            if i < self.servo_count:
                self.servo_angles[i] = value
                self.angle_sliders[i].set(value)

    def request_all_data(self):
        self.motor_speed_request_pub.publish(Int16MultiArray(data=[]))
        self.motor_position_request_pub.publish(Int16MultiArray(data=[]))
        self.servo_angle_request_pub.publish(Int16MultiArray(data=[]))

    def apply_motor_data(self):
        self.motor_speeds = [slider.get() for slider in self.speed_sliders]
        self.motor_positions = [slider.get() for slider in self.position_sliders]
        self.motor_speed_pub.publish(Int16MultiArray(data=self.motor_speeds))
        self.motor_position_pub.publish(Int16MultiArray(data=self.motor_positions))
        print("Applied Motor Speeds:", self.motor_speeds)
        print("Applied Motor Positions:", self.motor_positions)
    def apply_servo_data(self):
        self.servo_angles = [slider.get() for slider in self.angle_sliders]
        self.servo_angle_pub.publish(Int16MultiArray(data=self.servo_angles))
        print("Applied Servo Angles:", self.servo_angles)
if __name__ == "__main__":
    root = tk.Tk()
    app = MotorControlGUI(root)

    def ros_spin():
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    thread = threading.Thread(target=ros_spin)
    thread.daemon = True
    thread.start()

    root.mainloop()
