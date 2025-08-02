# yomi_driving/teleop.py

import roslaunch

def start_keyboard_teleop():
    # 키보드로 로봇을 수동 조종하는 텔레옵 노드를 실행하는 함수
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = roslaunch.rlutil.resolve_launch_arguments(['turtlebot3_teleop', 'turtlebot3_teleop_key.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [path])

def start_joystick_teleop():
    # 조이스틱으로 로봇을 수동 조종하는 텔레옵 노드를 실행하는 함수
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    path = roslaunch.rlutil.resolve_launch_arguments(['logitech_f710_joy_ros', 'joy_teleop.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [path])
