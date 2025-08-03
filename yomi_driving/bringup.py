# yomi_driving/bringup.py
import rospy
import roslaunch
     # TurtleBot3 로봇 구동을 위한 launch 파일 실행 함수
     # (모터, 센서, 통신 등 하드웨어 초기화)
def start_robot_bringup():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = roslaunch.rlutil.resolve_launch_arguments(['turtlebot3_bringup', 'turtlebot3_robot.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [path])

if __name__ == '__main__':
    rospy.init_node('robot_bringup_launcher', anonymous=True)

    bringup = start_robot_bringup()
    bringup.start()
    rospy.loginfo("TurtleBot3 bringup launched.")

    try:
        rospy.spin()  # 노드가 계속 실행되도록 유지
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down bringup...")
        bringup.shutdown()
