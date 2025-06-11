#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import tf.transformations
import math

class ObjectToGoal:
    def __init__(self):
        rospy.init_node('object_to_goal_node')
        self.sub = rospy.Subscriber('/detected_object', String, self.callback)
        self.pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.sent_labels = set()  # 중복 이동 방지

        # 객체별 목표 위치 정의 (x, y, yaw_deg)
        self.target_map = {
            "person": (1.0, 2.0, 90),
            "bottle": (2.5, 0.5, 180),
            "cup": (0.5, -1.0, 0)
        }

        rospy.loginfo("object_to_goal_node started.")
        rospy.spin()

    def callback(self, msg):
        label = msg.data.strip()
        if label in self.target_map and label not in self.sent_labels:
            x, y, yaw_deg = self.target_map[label]
            self.send_goal(x, y, yaw_deg)
            self.sent_labels.add(label)

    def send_goal(self, x, y, yaw_deg):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        yaw_rad = math.radians(yaw_deg)
        quat = tf.transformations.quaternion_from_euler(0, 0, yaw_rad)

        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]

        self.pub.publish(goal)
        rospy.loginfo(f"Sent goal for label → {x:.2f}, {y:.2f}, {yaw_deg:.1f}°")

if __name__ == '__main__':
    try:
        ObjectToGoal()
    except rospy.ROSInterruptException:
        pass

