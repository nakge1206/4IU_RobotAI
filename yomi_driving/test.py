# yomi_driving/main.py

from move import move_robot  # ✅ 함수 임포트

def main():
    print("[✓] 로봇 전진 테스트 시작")
    move_robot(linear_x=0.2, angular_z=0.0, duration=3.0)  # 3초간 전진
    print("[✓] 이동 완료")

if __name__ == "__main__":
    main()
