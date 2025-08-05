# yomi_driving/slam.py

import roslaunch
import rospy

def start_slam():
    # 실시간 SLAM (동시적 지도작성 및 위치추정)을 실행하는 함수
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = roslaunch.rlutil.resolve_launch_arguments(['turtlebot3_slam', 'turtlebot3_slam.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [path])

# ─────────────────────────────
# 단독 실행용 main
# ─────────────────────────────
if __name__ == "__main__":
    rospy.init_node("slam_launcher", anonymous=True)
    slam = start_slam()
    slam.start()
    rospy.loginfo("📍 SLAM 노드 실행됨 (Ctrl+C로 종료)")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("🛑 SLAM 종료 요청됨")
        slam.shutdown()
