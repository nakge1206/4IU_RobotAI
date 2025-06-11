#!/usr/bin/env python3

import rospy
import tkinter as tk
import threading
from std_msgs.msg import Int16MultiArray
from PIL import Image, ImageTk  # 이미지 처리 라이브러리

class MotorControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOMI Motor Control")
        self.root.geometry("1920x1080")  # 창 크기 조정

        rospy.init_node('gui_publisher', anonymous=True)

        # 배경 이미지 로드
        self.load_background()

        # 퍼블리셔 (명령 전송)
        self.motor_speed_pub = rospy.Publisher('/motor_speed_cmd', Int16MultiArray, queue_size=10)
        self.motor_position_pub = rospy.Publisher('/motor_position_cmd', Int16MultiArray, queue_size=10)
        self.servo_angle_pub = rospy.Publisher('/servo_angle_cmd', Int16MultiArray, queue_size=8)

        self.motor_speed_request_pub = rospy.Publisher('/motor_speed_request', Int16MultiArray, queue_size=10)
        self.motor_position_request_pub = rospy.Publisher('/motor_position_request', Int16MultiArray, queue_size=10)
        self.servo_angle_request_pub = rospy.Publisher('/servo_angle_request', Int16MultiArray, queue_size=8)


        # 데이터 저장 및 개수 설정
        self.motor_count = 10
        self.servo_count = 8
        self.motor_speeds = [0] * self.motor_count  
        self.motor_positions = [0] * self.motor_count  
        self.servo_angles = [0] * self.servo_count  

        # 슬라이더 개별 위치 설정
        self.set_slider_positions()

        # UI 요소 추가
        self.create_widgets()

        # ROS 구독 (피드백)
        rospy.Subscriber('/motor_speed_fb', Int16MultiArray, self.update_motor_speeds)
        rospy.Subscriber('/motor_position_fb', Int16MultiArray, self.update_motor_positions)
        rospy.Subscriber('/servo_angle_fb', Int16MultiArray, self.update_servo_angles)
    
    def load_background(self):
        """창 크기에 맞게 배경 이미지를 로드하고 적용"""
        try:
            image_path = "/home/micca/catkin_ws/src/yomi/scripts/image.png"
            image = Image.open(image_path)
            image = image.resize((1920, 1080))  # 창 크기에 맞게 조정
            self.bg_image = ImageTk.PhotoImage(image)

            # Canvas에 배경 이미지 추가
            self.canvas = tk.Canvas(self.root, width=1920, height=1080)
            self.canvas.pack(fill="both", expand=True)  # 창 크기에 맞게 확장
            self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image)  # 좌측 상단 기준 배치
        except Exception as e:
            print(f"배경 이미지 로드 실패: {e}")

    def set_slider_positions(self):
        """모터 속도, 위치, 서보 슬라이더의 개별 좌표를 설정"""
        self.speed_slider_positions = [
            (813, 595), (702, 555), (516, 555), (400, 555), 
            (1120, 595), (1238, 555), (1424, 555), (1540, 555),
            (952, 397),(1050, 207)
        ]
        
        self.position_slider_positions = [
            (813, 595+40), (702, 555+40), (516, 555+40), (400, 555+40), 
            (1120, 595+40), (1238, 555+40), (1424, 555+40), (1540, 555+40),
            (952, 397+40),(1050, 207+40)
        ]

        self.servo_slider_positions = [
            (213, 600),(213, 600+40),(213, 600+80),(213, 600+120),
            (1700, 600),(1700, 600+40),(1700, 600+80),(1700, 600+120)
        ]

    def create_widgets(self):
        """모터 및 서보 컨트롤 UI 생성"""

        # Load Data 버튼 (중앙 상단)
        self.load_button = tk.Button(self.root, text="Load Data", command=self.request_all_data)
        self.canvas.create_window(1920 // 2, 850, window=self.load_button)

        # 모터 속도 및 위치 슬라이더 추가
        self.speed_sliders = []
        self.position_sliders = []
        for i in range(self.motor_count):
            speed_slider = tk.Scale(self.root, from_=0, to=255, orient="horizontal")
            self.canvas.create_window(*self.speed_slider_positions[i], window=speed_slider)
            self.speed_sliders.append(speed_slider)

            position_slider = tk.Scale(self.root, from_=-100, to=100, orient="horizontal")
            self.canvas.create_window(*self.position_slider_positions[i], window=position_slider)
            self.position_sliders.append(position_slider)

        # "Apply" 버튼 추가 (모터)
        self.apply_motor_button = tk.Button(self.root, text="Apply Motor Speed & Position", command=self.apply_motor_data)
        self.canvas.create_window(1920 // 2, 900, window=self.apply_motor_button)

        # 서보 슬라이더 추가
        self.angle_sliders = []
        for i in range(self.servo_count):
            angle_slider = tk.Scale(self.root, from_=0, to=180, orient="horizontal")
            self.canvas.create_window(*self.servo_slider_positions[i], window=angle_slider)
            self.angle_sliders.append(angle_slider)

        # "Apply" 버튼 추가 (서보)
        self.apply_servo_button = tk.Button(self.root, text="Apply Servo Angles", command=self.apply_servo_data)
        self.canvas.create_window(1920 // 2, 950, window=self.apply_servo_button)

    def update_motor_speeds(self, msg):
        for i, value in enumerate(msg.data):
            if i < len(self.speed_sliders):
                self.speed_sliders[i].set(value)

    def update_motor_positions(self, msg):
        for i, value in enumerate(msg.data):
            if i < len(self.position_sliders):
                self.position_sliders[i].set(value)

    def update_servo_angles(self, msg):
        for i, value in enumerate(msg.data):
            if i < len(self.angle_sliders):
                self.angle_sliders[i].set(value)

    def request_all_data(self):
        self.motor_speed_request_pub.publish(Int16MultiArray(data=[]))
        self.motor_position_request_pub.publish(Int16MultiArray(data=[]))
        self.servo_angle_request_pub.publish(Int16MultiArray(data=[]))

    def apply_motor_data(self):
        motor_speeds = [slider.get() for slider in self.speed_sliders]
        motor_positions = [slider.get() for slider in self.position_sliders]
        self.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
        self.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
        print("Applied Motor Speeds:", motor_speeds)
        print("Applied Motor Positions:", motor_positions)

    def apply_servo_data(self):
        servo_angles = [slider.get() for slider in self.angle_sliders]
        self.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))
        print("Applied Servo Angles:", servo_angles)

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
