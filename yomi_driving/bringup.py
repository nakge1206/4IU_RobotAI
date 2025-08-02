# yomi_driving/bringup.py

import roslaunch
     # TurtleBot3 로봇 구동을 위한 launch 파일 실행 함수
     # (모터, 센서, 통신 등 하드웨어 초기화)
def start_robot_bringup():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = roslaunch.rlutil.resolve_launch_arguments(['turtlebot3_bringup', 'turtlebot3_robot.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [path])
