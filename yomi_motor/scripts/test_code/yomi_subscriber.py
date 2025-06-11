#!/usr/bin/env python

import rospy
from std_msgs.msg import Int16MultiArray

def received_speed_callback(msg):
    rospy.loginfo(f"Received Motor Speeds: {msg.data}")

def received_position_callback(msg):
    rospy.loginfo(f"Received Motor Positions: {msg.data}")

def received_servo_callback(msg):
    rospy.loginfo(f"Received Servo Angles: {msg.data}")
    rospy.loginfo(f"")

def received_data_listener():
    rospy.init_node('received_data_listener', anonymous=True)

    rospy.Subscriber('/received_motor_speed', Int16MultiArray, received_speed_callback)
    rospy.Subscriber('/received_motor_position', Int16MultiArray, received_position_callback)
    rospy.Subscriber('/received_servo_angle', Int16MultiArray, received_servo_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        received_data_listener()
    except rospy.ROSInterruptException:
        pass

