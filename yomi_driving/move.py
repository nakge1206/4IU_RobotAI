#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import math

class DistanceMover:
    def __init__(self, speed=0.2):
        """
        로봇 이동 제어 클래스
        - speed: 선속도(m/s) 또는 각속도(rad/s)
        """
        if not rospy.core.is_initialized():
            rospy.init_node('distance_mover', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        self.twist = Twist()
        self.speed = speed

    def send_motion(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        """지정된 시간 동안 속도 명령 전송"""
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
            self.twist.linear.x = linear_x
            self.twist.angular.z = angular_z
            self.pub.publish(self.twist)
            self.rate.sleep()
        # 정지
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.pub.publish(self.twist)

    def move_by_distance(self, forward=0.0, backward=0.0, left=0.0, right=0.0):
        """거리 또는 각도로 로봇 이동"""
        if forward > 0.0:
            duration = forward / self.speed
            self.send_motion(linear_x=+self.speed, duration=duration)

        if backward > 0.0:
            duration = backward / self.speed
            self.send_motion(linear_x=-self.speed, duration=duration)

        if left > 0.0:
            angular_speed = self.speed
            duration = (left * math.pi / 180) / angular_speed
            self.send_motion(angular_z=+angular_speed, duration=duration)

        if right > 0.0:
            angular_speed = self.speed
            duration = (right * math.pi / 180) / angular_speed
            self.send_motion(angular_z=-angular_speed, duration=duration)

    def move_forward(self, distance):
        """앞으로만 이동"""
        if distance > 0:
            duration = distance / self.speed
            self.send_motion(linear_x=+self.speed, duration=duration)
    
    def move_backward(self, distance):
        """뒤로만 이동"""
        if distance > 0:
            duration = distance / self.speed
            self.send_motion(linear_x=-self.speed, duration=duration)

    def rotate_in_place(self, angle_deg):
        """제자리에서 회전 (양수=왼쪽, 음수=오른쪽)"""
        angular_speed = 1.5  # rad/s
        duration = (abs(angle_deg) * math.pi / 180) / angular_speed
        if angle_deg > 0:
            self.send_motion(angular_z=+angular_speed, duration=duration)
        elif angle_deg < 0:
            self.send_motion(angular_z=-angular_speed, duration=duration)


if __name__ == "__main__":
    try:
        mover = DistanceMover(speed=0.2)

        rospy.sleep(1)
        rospy.loginfo("앞으로 1m 이동")
        mover.move_forward(1.0)

        rospy.sleep(1)
        rospy.loginfo("제자리에서 왼쪽으로 90도 회전")
        mover.rotate_in_place(90)

        rospy.sleep(1)
        rospy.loginfo("제자리에서 오른쪽으로 45도 회전")
        mover.rotate_in_place(-45)

    except rospy.ROSInterruptException:
        pass
