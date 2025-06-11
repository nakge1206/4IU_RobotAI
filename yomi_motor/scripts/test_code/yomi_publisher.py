#!/usr/bin/env python

import rospy
from std_msgs.msg import Int16

class YomiMotorController:
    def __init__(self):
        rospy.init_node('yomi_publisher', anonymous=True)

        # 퍼블리셔 생성 (개별 토픽으로 수정)
        self.speed_pubs = [rospy.Publisher(f'/motor_{i+1}/speed', Int16, queue_size=10) for i in range(10)]
        self.position_pubs = [rospy.Publisher(f'/motor_{i+1}/position', Int16, queue_size=10) for i in range(10)]
        self.servo_pubs = [rospy.Publisher(f'/servo_{i+1}/angle', Int16, queue_size=8) for i in range(8)]

        # 초기값 설정
        self.speed_values = [30, 40, 50, 60, 70, 80, 90, 20, 10, 100]  # DC 모터 속도 (0~100)
        self.position_values = [0, 45, 90, 135, 180, 225, 270, 315, 360, 180]  # DC 모터 각도 (0~360)
        self.servo_values = [0, 30, 60, 90, 120, 150, 180, 90]  # 서보 모터 각도 (0~180)

        # 서브스크라이버 등록 (개별 토픽으로 수정)
        for i in range(10):
            rospy.Subscriber(f'/motor_{i+1}/speed', Int16, self.create_motor_speed_callback(i))
            rospy.Subscriber(f'/motor_{i+1}/position', Int16, self.create_motor_position_callback(i))
        
        for i in range(8):
            rospy.Subscriber(f'/servo_{i+1}/angle', Int16, self.create_servo_angle_callback(i))

    def create_motor_speed_callback(self, index):
        def callback(msg):
            self.speed_values[index] = msg.data
        return callback

    def create_motor_position_callback(self, index):
        def callback(msg):
            self.position_values[index] = msg.data
        return callback

    def create_servo_angle_callback(self, index):
        def callback(msg):
            self.servo_values[index] = msg.data
        return callback

    def publish_data(self):
        """ 주기적으로 데이터 발행 """
        rate = rospy.Rate(10)  # 10Hz

        while not rospy.is_shutdown():
            for i in range(10):
                speed_msg = Int16()
                speed_msg.data = self.speed_values[i]
                self.speed_pubs[i].publish(speed_msg)

                position_msg = Int16()
                position_msg.data = self.position_values[i]
                self.position_pubs[i].publish(position_msg)

            for i in range(8):
                servo_msg = Int16()
                servo_msg.data = self.servo_values[i]
                self.servo_pubs[i].publish(servo_msg)

            rospy.loginfo(f"Publishing Motor Speeds: {self.speed_values}")
            rospy.loginfo(f"Publishing Motor Positions: {self.position_values}")
            rospy.loginfo(f"Publishing Servo Angles: {self.servo_values}")
            rospy.loginfo(" ")  # 줄 바꿈

            rate.sleep()  # CPU 과부하 방지

if __name__ == '__main__':
    try:
        motor_controller = YomiMotorController()
        motor_controller.publish_data()
    except rospy.ROSInterruptException:
        pass