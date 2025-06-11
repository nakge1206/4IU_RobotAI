#!/usr/bin/env python3

import rospy
import tkinter as tk
import threading
import time  # 추가: 타임스탬프용
from std_msgs.msg import Int16MultiArray, Bool
from PIL import Image, ImageTk  # 이미지 처리 라이브러리
from tkinter import filedialog  # 추가: 파일 선택 대화상자용

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
        self.motor_speeds = [1] * self.motor_count  
        self.motor_positions = [180] * self.motor_count  
        self.servo_angles = [90] * self.servo_count  

        self.servo_ids = [1, 2, 3, 4, 6, 7, 8, 9]

        # 슬라이더 개별 위치 설정
        self.set_slider_positions()

        # UI 요소 추가 (기존 모터/서보 관련 UI)
        self.create_widgets()

        # ROS 구독 (피드백)
        rospy.Subscriber('/motor_speed_fb', Int16MultiArray, self.update_motor_speeds)
        rospy.Subscriber('/motor_position_fb', Int16MultiArray, self.update_motor_positions)
        rospy.Subscriber('/servo_angle_fb', Int16MultiArray, self.update_servo_angles)
        
        # ───────────────────────────────────────────────
        #  4개의 스위치 관련 토픽 Sub, 버튼 생성
        self.create_switch_widgets()
        self.subscribe_switch_topics()
        # ───────────────────────────────────────────────

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
            (813, 615+62), (702, 575+62), (516, 575+62), (400, 575+62), 
            (1120, 615+62), (1238, 575+62), (1424, 575+62), (1540, 575+62),
            (952, 417+62), (1050, 187+62)
        ]
        
        self.position_slider_positions = [
            (813, 615), (702, 575), (516, 575), (400, 575), 
            (1120, 615), (1238, 575), (1424, 575), (1540, 575),
            (952, 417), (1050, 187)
        ]

        self.servo_slider_positions = [
            (213, 600), (213, 600+62), (213, 600+124), (213, 600+186),
            (1700, 600), (1700, 600+62), (1700, 600+124), (1700, 600+186)
        ]
        
        # 각 position_slider의 최소/최대 범위를 개별적으로 지정 (총 motor_count 개)
        self.position_slider_ranges = [
            (120, 300),(100, 260),(90, 270),(95, 180),
            (60, 240),(100, 260),(90, 270),(180, 265),   # Motor 10
            (120, 240),(110, 250)
        ]

    def create_widgets(self):
        """모터 및 서보 컨트롤 UI 생성"""

        # Load Data 버튼 (중앙 상단)
        self.load_button = tk.Button(
            self.root,
            text="Load Data",
            command=self.request_all_data,
            font=("Arial", 22, "bold"),
            width=15,
            height=1,
            bd=5,
            highlightthickness=2,
            relief=tk.RIDGE
        )
        self.canvas.create_window(1920 // 2, 830, window=self.load_button)

        # 모터 속도 및 위치 슬라이더와 라벨 추가
        self.speed_sliders = []
        self.position_sliders = []
        self.speed_labels = []
        self.position_labels = []
        
        for i in range(self.motor_count):
            # 속도 라벨 (슬라이더 위에 배치)
            speed_label = tk.Label(self.root, text=f"Motor {i+3} Speed", fg="white", bg="black")
            self.canvas.create_window(self.speed_slider_positions[i][0], self.speed_slider_positions[i][1] - 30, window=speed_label)
            self.speed_labels.append(speed_label)

            # 속도 슬라이더
            speed_slider = tk.Scale(
                self.root,
                from_=0, to=100,
                orient="horizontal", bg="black", troughcolor="white", fg="white",
                command=lambda val, idx=i: self.update_label(idx, val, "speed")
            )
            self.canvas.create_window(*self.speed_slider_positions[i], window=speed_slider)
            speed_slider.set(self.motor_speeds[i])
            self.speed_sliders.append(speed_slider)

            # 위치 라벨 (슬라이더 위에 배치)
            position_label = tk.Label(self.root, text=f"Motor {i+3} Position", fg="white", bg="gray")
            self.canvas.create_window(self.position_slider_positions[i][0], self.position_slider_positions[i][1] - 30, window=position_label)
            self.position_labels.append(position_label)

            # 위치 슬라이더 (개별 min, max 값 지정)
            pos_range = self.position_slider_ranges[i]
            position_slider = tk.Scale(
                self.root,
                from_=pos_range[0], to=pos_range[1],
                orient="horizontal", bg="gray", troughcolor="white", fg="white",
                command=lambda val, idx=i: self.update_label(idx, val, "position")
            )
            self.canvas.create_window(*self.position_slider_positions[i], window=position_slider)
            position_slider.set(self.motor_positions[i])
            self.position_sliders.append(position_slider)

        # "Apply" 버튼 추가 (모터)
        self.apply_motor_button = tk.Button(
            self.root, 
            text="Apply Motor Speed & Position", 
            command=self.apply_motor_data,
            font=("Arial", 22, "bold"),
            width=30,
            height=1,
            bd=5,
            highlightthickness=2,
            relief=tk.RIDGE
        )
        self.canvas.create_window(1920 // 2, 900, window=self.apply_motor_button)

        # 서보 슬라이더와 라벨 추가
        self.angle_sliders = []
        self.servo_labels = []
        for i in range(self.servo_count):
            servo_label = tk.Label(self.root, text=f"Servo {self.servo_ids[i]} Angle", fg="white", bg="blue")
            self.canvas.create_window(self.servo_slider_positions[i][0], self.servo_slider_positions[i][1] - 30, window=servo_label)
            self.servo_labels.append(servo_label)

            angle_slider = tk.Scale(
                self.root,
                from_=0, to=180,
                orient="horizontal", fg="white", bg="blue",
                command=lambda val, idx=i: self.update_label(idx, val, "servo")
            )
            self.canvas.create_window(*self.servo_slider_positions[i], window=angle_slider)
            angle_slider.set(self.servo_angles[i])
            self.angle_sliders.append(angle_slider)

        # "Apply" 버튼 추가 (서보)
        self.apply_servo_button = tk.Button(
            self.root, 
            text="Apply Servo Angles", 
            command=self.apply_servo_data,
            font=("Arial", 22, "bold"),
            width=20,
            height=1,
            bd=5,
            highlightthickness=2,
            relief=tk.RIDGE
        )
        self.canvas.create_window(1920 // 2, 970, window=self.apply_servo_button)
        
        # ───────────────────────────────────────────────
        # [추가] "Save Motion" 버튼 추가 - 모션 데이터를 파일로 저장
        self.save_motion_button = tk.Button(
            self.root,
            text="Save Motion",
            command=self.save_motion,
            font=("Arial", 22, "bold"),
            width=20,
            height=1,
            bd=5,
            highlightthickness=2,
            relief=tk.RIDGE
        )
        self.canvas.create_window(1600, 900, window=self.save_motion_button)
        
        # [추가] "Load Motion" 버튼 추가 - 파일에서 모션 데이터를 읽어 슬라이더 업데이트
        self.load_motion_button = tk.Button(
            self.root,
            text="Load Motion",
            command=self.load_motion_file,
            font=("Arial", 22, "bold"),
            width=20,
            height=1,
            bd=5,
            highlightthickness=2,
            relief=tk.RIDGE
        )
        self.canvas.create_window(1600, 970, window=self.load_motion_button)

    def update_label(self, index, value, slider_type):
        """슬라이더 값을 변경하면 라벨도 업데이트"""
        if slider_type == "speed":
            self.speed_labels[index].config(text=f"Motor {index+3} Speed")
        elif slider_type == "position":
            self.position_labels[index].config(text=f"Motor {index+3} Position")
        elif slider_type == "servo":
            self.servo_labels[index].config(text=f"Servo {self.servo_ids[index]} Angle")

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
        
    # ───────────────────────────────────────────────
    # [추가] 스위치 버튼 및 토픽 처리 관련 함수
    def create_switch_widgets(self):
        """4개의 스위치 버튼 생성 (추가)"""
        self.switch_buttons = []
        positions = [(1400, 50), (1500, 50), (1600, 50), (1700, 50)]
        labels = ["Sw 1", "Sw 2", "Sw 3", "Sw 4"]
        for i in range(4):
            btn = tk.Button(
                self.root,
                text=labels[i],
                font=("Arial", 18, "bold"),
                width=4,
                height=1,
                bd=5,
                bg="gray",  # 기본 색상 (OFF 상태)
                relief=tk.RIDGE
            )
            self.canvas.create_window(positions[i][0], positions[i][1], window=btn)
            self.switch_buttons.append(btn)

    def subscribe_switch_topics(self):
        """4개의 스위치 토픽 구독 설정 (추가)"""
        self.switch_subs = []
        for i in range(4):
            topic = f"/switch_{i+1}_state"
            sub = rospy.Subscriber(topic, Bool, self.update_switch_state, callback_args=i)
            self.switch_subs.append(sub)

    def update_switch_state(self, msg, index):
        """스위치 토픽에서 받은 값에 따라 버튼 색상 변경 (추가)"""
        if msg.data:
            self.switch_buttons[index].config(bg="green")
        else:
            self.switch_buttons[index].config(bg="gray")
    # ───────────────────────────────────────────────
    # [추가] 모션 데이터를 저장하는 함수
    def save_motion(self):
        """현재 모터 speed, position, 서보 각도 (총 28개 값)를 별도의 텍스트 파일로 저장"""
        motor_speeds = [slider.get() for slider in self.speed_sliders]
        motor_positions = [slider.get() for slider in self.position_sliders]
        servo_angles = [slider.get() for slider in self.angle_sliders]
        motion_data = motor_speeds + motor_positions + servo_angles
        # 파일 저장 경로를 지정하여 파일명 생성
        filename = "/home/micca/catkin_ws/src/yomi/motion/motion_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
        try:
            with open(filename, "w") as f:
                f.write(",".join(str(x) for x in motion_data))
            print(f"Motion data saved to {filename}")
        except Exception as e:
            print(f"Failed to save motion data: {e}")
    
            
    # [추가] 모션 데이터를 파일에서 읽어 슬라이더 업데이트하는 함수
    def load_motion_file(self):
        """파일 선택 대화상자를 통해 모션 데이터를 읽어와 슬라이더 값을 업데이트"""
        filename = filedialog.askopenfilename(
            title="Select Motion File",
            initialdir="/home/micca/catkin_ws/src/yomi/motion",  # 기본 폴더 설정
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if not filename:
            print("No file selected.")
            return
        try:
            with open(filename, "r") as f:
                data_str = f.read().strip()
            # 콤마로 구분된 문자열을 정수 리스트로 변환 (총 28개 값)
            values = [int(x) for x in data_str.split(",")]
            if len(values) != 28:
                print(f"Invalid motion data length: {len(values)}. Expected 28 values.")
                return
            # 모터 속도 (10), 모터 위치 (10), 서보 각도 (8)
            motor_speeds = values[0:10]
            motor_positions = values[10:20]
            servo_angles = values[20:28]
            # 슬라이더 값 업데이트
            for i in range(self.motor_count):
                self.speed_sliders[i].set(motor_speeds[i])
                self.position_sliders[i].set(motor_positions[i])
            for i in range(self.servo_count):
                self.angle_sliders[i].set(servo_angles[i])
            print(f"Motion data loaded from {filename}")
        except Exception as e:
            print(f"Failed to load motion data: {e}")
    

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
