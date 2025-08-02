# yomi_driving/slam.py

import roslaunch

def start_slam():
    # 실시간 SLAM (동시적 지도작성 및 위치추정)을 실행하는 함수
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = roslaunch.rlutil.resolve_launch_arguments(['turtlebot3_slam', 'turtlebot3_slam.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [path])