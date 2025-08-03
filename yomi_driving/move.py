import rospy
from geometry_msgs.msg import Twist

def move_robot(linear_x=0.2, angular_z=0.0, duration=2.0):
    rospy.init_node('manual_mover', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)

    twist = Twist()
    twist.linear.x = linear_x    # 전/후진 속도
    twist.angular.z = angular_z  # 좌/우 회전 속도

    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time).to_sec() < duration:
        pub.publish(twist)
        rate.sleep()

    # 멈춤 명령
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    pub.publish(twist)
