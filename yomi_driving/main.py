import rospy
import os

from bringup import start_robot_bringup
from teleop import start_joystick_teleop, start_keyboard_teleop
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

    # [2] 수동 주행 선택
    teleop = None
    teleop_choice = input("[2] 수동 주행 방식 선택 (k: 키보드 / j: 조이스틱 / s: 생략): ")
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

    # [3] 맵 파일 선택
    MAP_NAME = input("사용할 맵 이름을 입력하세요 (확장자 제외): ").strip()
    MAP_DIR = os.path.expanduser("~")
    map_path = os.path.join(MAP_DIR, f"{MAP_NAME}.yaml")

    if not os.path.exists(map_path):
        rospy.logerr(f"[!] 맵 파일을 찾을 수 없습니다: {map_path}")
        return

    # [4] 자율주행 시작
    print("[✓] 자율주행 시작 중...")
    try:
        nav = start_navigation(map_path)
        nav.start()
    except Exception as e:
        rospy.logerr(f"[!] 자율주행 실행 실패: {e}")
        return

    rospy.spin()

if __name__ == "__main__":
    main()
