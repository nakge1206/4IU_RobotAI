import rospy
from geometry_msgs.msg import Twist

def move_robot_by_distance(forward=0.0, backward=0.0, left=0.0, right=0.0, speed=0.2):
    """
    지정한 거리만큼 로봇을 이동시키는 함수
    - forward: 앞쪽 이동 거리 (m)
    - backward: 뒤쪽 이동 거리 (m)
    - left: 왼쪽 회전 거리 (deg, 각도 기반이므로 회전 반경 고려X)
    - right: 오른쪽 회전 거리 (deg)
    - speed: 선속도 또는 각속도 (m/s 또는 rad/s)
    """

    rospy.init_node('distance_mover', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)
    twist = Twist()

    def send_motion(linear_x=0.0, angular_z=0.0, duration=1.0):
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            pub.publish(twist)
            rate.sleep()
        # 정지
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

    # 직진
    if forward > 0.0:
        duration = forward / speed
        send_motion(linear_x=+speed, duration=duration)

    # 후진
    if backward > 0.0:
        duration = backward / speed
        send_motion(linear_x=-speed, duration=duration)

    # 왼쪽 회전 (deg → rad 변환 필요)
    if left > 0.0:
        angular_speed = speed  # rad/s
        duration = (left * 3.141592 / 180) / angular_speed
        send_motion(angular_z=+angular_speed, duration=duration)

    # 오른쪽 회전
    if right > 0.0:
        angular_speed = speed
        duration = (right * 3.141592 / 180) / angular_speed
        send_motion(angular_z=-angular_speed, duration=duration)

# 사용 예시:
# 앞으로 1m, 왼쪽으로 90도 회전
# move_robot_by_distance(forward=1.0, left=90, speed=0.2)
