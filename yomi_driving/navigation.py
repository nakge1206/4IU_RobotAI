# yomi_driving/navigation.py

import roslaunch

def start_navigation(map_path):
    # 저장된 맵(.yaml)을 기반으로 자율주행 노드를 실행하는 함수
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch_file = roslaunch.rlutil.resolve_launch_arguments(
        ['turtlebot3_navigation', 'turtlebot3_navigation.launch'])[0]
    return roslaunch.parent.ROSLaunchParent(uuid, [launch_file], args=[f'map_file:={map_path}'])
