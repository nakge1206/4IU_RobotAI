# yomi_driving/main.py

import rospy
import os

from bringup import start_robot_bringup
from teleop import start_joystick_teleop, start_keyboard_teleop
from slam import start_slam
from map_saver import save_map
from map_editor import open_kolourpaint
from navigation import start_navigation

def main():
    rospy.init_node('turtlebot3_autonomy_main', anonymous=True)

    print("[1] 로봇 Bringup 중...")
    try:
        bringup = start_robot_bringup()
        bringup.start()
        rospy.sleep(3)
    except Exception as e:
        rospy.logerr(f"[!] Bringup 실패: {e}")
        return

    print("[2] SLAM 시작...")
    try:
        slam = start_slam()
        slam.start()
        print("[✓] SLAM 시작됨")
    except Exception as e:
        rospy.logerr(f"[!] SLAM 실행 실패: {e}")
        return

    # [3] 수동 주행 선택
    teleop = None
    teleop_choice = input("[3] SLAM 중 수동 주행 방식 선택 (k: 키보드 / j: 조이스틱 / s: 생략): ")
    try:
        if teleop_choice == 'k':
            teleop = start_keyboard_teleop()
            teleop.start()
        elif teleop_choice == 'j':
            teleop = start_joystick_teleop()
            teleop.start()
        else:
            print("[i] 수동 주행 생략됨")
    except Exception as e:
        rospy.logwarn(f"[!] Teleop 실행 실패: {e}")

    # [4] SLAM 진행 후 사용자 입력 대기
    input("맵핑이 완료되면 Enter를 누르세요...")

    print("[5] 맵 저장 중...")
    save_map("mapgodd")

    # [6] 맵 편집 여부
    edit = input("맵을 편집하려면 'y' 입력: ")
    if edit.lower() == 'y':
        open_kolourpaint()
        input("편집 완료 후 Enter...")

    # [7] SLAM, Teleop 종료
    slam.shutdown()
    if teleop:
        teleop.shutdown()
    rospy.sleep(2)

    # [8] 자율주행 시작
    print("[6] 자율주행 시작 중...")
    MAP_NAME = "mapgodd"
    MAP_DIR = os.path.expanduser("~")
    map_path = os.path.join(MAP_DIR, f"{MAP_NAME}.yaml")
    try:
        nav = start_navigation(map_path)
        nav.start()
    except Exception as e:
        rospy.logerr(f"[!] 자율주행 실행 실패: {e}")
        return

    rospy.spin()

if __name__ == "__main__":
    main()
