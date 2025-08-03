import rospy
from std_msgs.msg import String, Int16MultiArray
from yomi_motor import MotionSequenceExecutor

class MotionController:
    def __init__(self, vision = None):
        rospy.init_node('motion_controller_node', anonymous=True)
        self.executor = MotionSeq uenceExecutor()
        self.vision = vision

    def stand(self):
        motor_speeds = [3] * 10
        motor_positions = [180, 110, 180, 180, 180, 250, 180, 180, 190, 180]
        servo_angles = [90] * 8

        self.executor.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
        self.executor.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
        self.executor.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))

    def ifConflicting(self):
        pass
    
    def I_joy1(self):
        """(좋아하는 것에 대해) 팔을 살랑살랑 움직인다."""
        # self.executor.request_all_data()
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[180], motor_speeds=[5])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[180], motor_speeds=[5])
        rospy.sleep(0.2)
        self.stand()
        rospy.sleep(0.1)
        for i in range(3):
            self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[160, 160], motor_speeds=[4, 4])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[200, 200], motor_speeds=[4, 4])
            rospy.sleep(1.5)

    def I_joy2(self):
        "몸을 좌우로 조금씩 움직인다."

    def I_joy3(self):
        "가까운 사용자에게 다가간다."

    def I_trust1(self):
        "신뢰하는 사용자 옆에서 따라다님"

    def I_trust2(self):
        "사용자 지시가 있으면 즉각 응답 및 행동을 수행"

    def I_trust3(self):
        "행동 및 이벤트 뒤 사용자 쪽으로 돌아옴"

    def I_fear1(self):
        "(위협 감지시)모서리로 움직인다 벽을 보고 선다"

    def I_fear2(self):
        "신뢰하는 사용자 뒤로 숨는다"

    def I_fear3(self):
        "큰 물건 뒤로 숨는다"

    def I_surprise1(self):
        "(소리-(어? 뭐지?) 작은 목소리 출력 후) 바퀴가 뒤로 살짝 밀리며 정지"

    def I_surprise2(self):
        "미세 움직임으로 진동 표현, 떨림 표현"
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[5])
        rospy.sleep(0.2)
        for i in range(3):
            self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[90, 270], motor_speeds=[4, 4])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[200, 160], motor_speeds=[4, 4])
            rospy.sleep(1.5)

    def I_surprise3(self):
        "((호불호에서 호의 경우) 대상으로부터 가까워지고, (불호의 경우) 대상으로부터 멀어짐)"

    def I_sadness1(self):
        "느리게 이동"

    def I_sadness2(self):
        "고개를 숙이듯 머리가 아래로 향함"
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[159], motor_speeds=[4])
        rospy.sleep(1.5)

    def I_sadness3(self):
        "일정 시간 정지 상태 유지"
        self.stand()
        rospy.sleep(1.5)

    def I_disgust1(self):
        "원치 않는 사용자 접근 시 후진"

    def I_disgust2(self):
        "회피 루트를 탐색"
    
    def I_disgust3(self):
        "도망, 회피 , 뒤 돌기"

    def I_anger1(self):
        "본체의 바퀴가 빠르게 앞뒤로 움직임"

    def I_anger2(self):
        "몸 전체에 진동(가능하면)"
        for i in range(5):
            self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7])
            self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[98, 262], motor_speeds=[7])

    def I_anticipation1(self):
        "주변을 돌아다닌다"

    def I_anticipation2(self):
        "바퀴로 자리에서 빙글빙글 천천히 돈다"
    
    def I_anticipation3(self):
        "새로운 입력을 기다리는 듯 정지 후 전방 주시"
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[220], motor_speeds=[4])

    def E_joy1(self):
        "(행위가 끝난뒤) 크게 공간을 한바퀴 돌며 정해진 위치로 돌아감"
    
    def E_joy2(self):
        "정위치에서 양손을 든다."
        self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[300, 60], motor_speeds=[7])
        rospy.sleep(1.5)

    def E_joy3(self):
        "바퀴로 신나게 빙빙 돈다"

    def E_trust1(self):
        "신뢰대상에게 일정거리 유지하며 옆에 선다."
    
    def E_trust2(self):
        "일정거리 유지하며 따라다닌다."

    def E_trust3(self):
        "사용자 지시가 있으면 즉각 응답 및 행동을 수행하고 대기한다."
    
    def E_fear1(self):
        "(위협감지 시)모서리로 움직인다 벽을 보고 선다."

    def move_to_greeting_pose(self):
        # 5번 모터 위치로 팔을 올리고 서보로 손 흔들기
        self.executor.motor_publisher(motor_id=5, motor_position=320)
        self.executor.motor_publisher(servo_id=6, servo_angle=120)
    
    def bow(self):
        self.executor.motor_publisher(motor_id=7, motor_position=400)
        self.executor.motor_publisher(servo_id=4, servo_angle=45)

    def run(self):
        # 예시: 반복 동작
        rate = rospy.Rate(0.2)  # 5초마다
        while not rospy.is_shutdown():
            self.move_to_greeting_pose()
            rospy.sleep(2)
            self.bow()
            rospy.sleep(2)
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MotionController()
        controller.I_joy1()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass