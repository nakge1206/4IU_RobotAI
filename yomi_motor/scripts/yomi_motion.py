import os
import sys

# 1단계: 현재 파일 경로
#current_dir = os.path.dirname(os.path.abspath(__file__))

# 2단계: 두 단계 상위 폴더
#base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 3단계: yomi_driving 폴더 경로 추가
#yomi_driving_path = os.path.join(base_dir, "yomi_driving")
#sys.path.append(yomi_driving_path)

current_dir = os.path.dirname(os.path.abspath(__file__))                    # .../yomi_motor/scripts
base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))          # .../4IU_RobotAI

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import rospy
from std_msgs.msg import Int16MultiArray
from yomi_motor.yomi_motor_run import MotionSequenceExecutor
from yomi_driving.move import DistanceMover
from yomi_driving.automove import ObstacleAvoider

class MotionController:
    def __init__(self, vision = None):
        if not rospy.core.is_initialized():
            rospy.init_node('motion_controller_node', anonymous=True)
        self.executor = MotionSequenceExecutor()
        self.vision = vision
        self.mover = DistanceMover(speed=2.0)
        self.automover = ObstacleAvoider()
        self.vision_detect = []
        self.vision_location = []
    
    def _vision_information(self):
        # detects = self.vision.vision_get_detections()
        detects = None
        self.vision_detect.clear()
        self.vision_location.clear()
        if detects is not None:
            for item in detects:
                self.vision_detect.append(item['label'])
                box = item['box']
                center_x = (box[0] + box[2])/2
                center_y = (box[1] + box[3]) / 2
                self.vision_location.append((center_x, center_y))

    def frontal(self):
        # 로봇이 정면을 바라볼 때 우선 detect 되는 애가 있으면 발동을 하는 것이기에 이것은 정면을 봐야하는 다른 동작이 있을 시 그 동작 내에 if문을 추가해서 이 함수를 발동 시키게 하는게 맞는 거 같다.
        # 로봇은 우선 좌표를 받아왔으니 정면을 보게 한다. 그 이후 이미 저장된 좌표로 움직이면 되니

        max_attempts = 10
        attempts = 0

        # 정확한 값 말고 오차범위 상정
        while attempts < max_attempts:
            self._vision_information()
            if self.vision_location:
                coordinate_x = self.vision_location[0][0]
            coordinate_x = None

            if coordinate_x is None:
                break
            if 260 <= coordinate_x <= 380:
                break

            if coordinate_x < 260:
                self.mover.rotate_in_place(5)
            elif coordinate_x > 380:
                self.mover.rotate_in_place(-5)
            rospy.sleep(0.3)
            attempts += 1

    def wait_command(self):
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[130], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[230], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[180], motor_speeds=[7])

    def finger_end(self):
        self.executor.motor_publisher_batch(servo_ids=[2], servo_angles=[120])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[2], servo_angles=[130])

    


    def stand(self):
        motor_speeds = [3] * 10
        motor_positions = [180, 110, 180, 180, 180, 250, 180, 180, 190, 180]
        servo_angles = [90] * 8

        self.executor.motor_speed_pub.publish(Int16MultiArray(data=motor_speeds))
        self.executor.motor_position_pub.publish(Int16MultiArray(data=motor_positions))
        self.executor.servo_angle_pub.publish(Int16MultiArray(data=servo_angles))

    def ifConflicting(self):
        pass

    def defalt_motion(self): # filiming_motion2로 하면 될 듯
        "작동 시 처음 해야하는 동작" """고개정면, 팔꿈치만 펴는거"""
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[180, 180], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[11, 12], motor_positions=[190, 180], motor_speeds=[6])
        rospy.sleep(0.2)

    def defalt_motion2(self):
        """팔만 차렸자세로 만드는거"""
        self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[180, 180], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[4, 8], motor_positions=[110, 250], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[180, 180], motor_speeds=[6])
        rospy.sleep(0.2)

    def defalt_motion3(self): 
        """손가락 주먹쥐게 만드는거"""
        self.executor.motor_publisher_batch(servo_ids=[1], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[2], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[3], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[4], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[6], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[7], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[8], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[9], servo_angles=[90])
        rospy.sleep(0.2)
    
    def test(self):
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7, 7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[220, 140], motor_speeds=[7, 7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[180, 180], motor_speeds=[7, 7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7, 7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[180, 180], motor_speeds=[7, 7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7, 7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[180, 180], motor_speeds=[7, 7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7, 7])
        rospy.sleep(1)

    """
        11, 12: 머리 위 아래, 좌우
        3, 7: 어깨 앞뒤로
        4, 8: 팔 옆으로
        5, 9: 전완근 회전
        6, 10: 전완근 들고 내리고
    """
    # 3, 7: 300 ~ 120 300이 위로 들기 120이 뒤로 뻗기 180 중앙, 60~ 240 60이 위로 240이 뒤로 뻗기 180 중앙
    # 4, 8: 110~ 260 110이 내리기 260이 들기, 100~250 250이 내리기 100이 들기
    # 5, 9: 90~ 270, 안쪽으로 굽는게 낮은 숫자 180 중앙, 90~270 안쪽으로 굽는게 높은 숫자 180 중앙 
    # 6, 10: 95~180 95가 굽은거 180이 핀거, 180 ~ 265 265가 굽은거 180이 핀거 
    # 11: 159가 아래인 걸 보니 숫자를 내리면 머리를 아래로 향함 180 중앙
    # 12: 130~230 180 중앙

    def homshowping1(self):
        """걸어나가는 팔행동"""
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7, 7])
        rospy.sleep(0.5)
        self.executor.motor_publisher_batch(motor_ids=[4, 8], motor_positions=[120, 240], motor_speeds=[7, 7])
        rospy.sleep(1)
        for i in range(4):
            if i % 2 != 0:
                self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[250, 230], motor_speeds=[9, 9]) 
                rospy.sleep(1.8)
            else: 
                self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[130, 110], motor_speeds=[9, 9])
                rospy.sleep(1.8)
        
    def Hi(self):
        self.executor.motor_publisher_batch(motor_ids=[4], motor_positions=[180], motor_speeds=[8])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5], motor_positions=[270], motor_speeds=[8])
        rospy.sleep(1)
        for i in range(7):
            if i%2 != 0:
                self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[150], motor_speeds=[9])
                rospy.sleep(0.7)
            else: 
                self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[100], motor_speeds=[9])
                rospy.sleep(0.7)

    def Hi2(self):
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[180], motor_speeds=[9]) 
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[4], motor_positions=[110], motor_speeds=[9]) 
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5], motor_positions=[180], motor_speeds=[9])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[180], motor_speeds=[9])

    def filming_motion1(self):
        """팔 뒤로 뺴는거 (홈쇼핑#5에 사용)"""
        self.executor.motor_publisher_batch(motor_ids=[11, 12], motor_positions=[190, 180], motor_speeds=[7]) # 팔이나 머리 움직일때는 이 코드 참조
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[141], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[214], motor_speeds=[7])
        # rospy.sleep(5)
        # self.defalt_motion()
        # self.defalt_motion2()

    def filiming_motion2(self):
        """차렷자세"""
        self.defalt_motion()
        self.defalt_motion2()
    
    def filiming_motion3(self):
        """팔을 만세하고, 제자리에서 한바퀴 돈 다음에, "안녕 난 요미야" 출력 / 근데 TTS는 아직 안함 (홈쇼핑#6,뉴스#1 사용)"""
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[300], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[60], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[4], motor_positions=[172], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[8], motor_positions=[188], motor_speeds=[7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[4], motor_positions=[140], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[8], motor_positions=[220], motor_speeds=[7])
        rospy.sleep(2)
        self.mover.rotate_in_place(360)
        # rospy.sleep(5)
        # self.defalt_motion()
        # self.defalt_motion2()

    def filiming_motion4(self):
        """목, 어깨, 팔, 손가락이 모두 움직이는 모양(홈쇼핑#10)"""
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[210], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[130], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[210], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5], motor_positions=[90], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[6])
        rospy.sleep(3)
        self.executor.motor_publisher_batch(servo_ids=[6], servo_angles=[60]) #손가락은 이거
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[7], servo_angles=[60])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[8], servo_angles=[60])
        # rospy.sleep(3)
        # self.stand()
    
    def filiming_motion5(self):
        """목, 어깨, 팔, 손가락이 모두 움직이는 모양(홈쇼핑#10)"""
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[210], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[130], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[210], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5], motor_positions=[90], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[6])
        rospy.sleep(3)
        self.executor.motor_publisher_batch(servo_ids=[6], servo_angles=[60]) #손가락은 이거
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[7], servo_angles=[60])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[8], servo_angles=[60])
        # rospy.sleep(3)
        # self.stand()

    def gababo_ba(self): # 가위바위보 요미가 묵
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[130], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
        rospy.sleep(1)
        for i in range(7):
            if i % 2 !=0:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[7])
                rospy.sleep(1)
            else:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
                rospy.sleep(1)
        self.executor.motor_publisher_batch(servo_ids=[6], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[7], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[8], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[9], servo_angles=[90])
        # rospy.sleep(3)
        

    def gababo_ga(self): # 가위바위보 요미가 가위
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[130], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
        rospy.sleep(1)
        for i in range(7):
            if i % 2 !=0:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[7])
                rospy.sleep(1)
            else:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
                rospy.sleep(1)
        self.executor.motor_publisher_batch(servo_ids=[6], servo_angles=[90])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[7], servo_angles=[0])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[8], servo_angles=[0])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[9], servo_angles=[90])
        rospy.sleep(3)
        self.defalt_motion3()

    def gababo_bo(self): # 가위바위보 요미가 보
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[130], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
        rospy.sleep(1)
        for i in range(7):
            if i % 2 !=0:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[7])
                rospy.sleep(1)
            else:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
                rospy.sleep(1)
        self.executor.motor_publisher_batch(servo_ids=[6], servo_angles=[0])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[7], servo_angles=[0])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[8], servo_angles=[0])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(servo_ids=[9], servo_angles=[0])
        rospy.sleep(3)
        self.defalt_motion3()

    def scenario_motion2_1(self): # 요미가 맛있는거 먹는다고 신나하는 장면
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[180], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[300], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[60], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[4], motor_positions=[172], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[8], motor_positions=[188], motor_speeds=[7])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[4], motor_positions=[140], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[8], motor_positions=[220], motor_speeds=[7])
        # rospy.sleep(2)
        # self.defalt_motion()
        # self.defalt_motion2()

    def scenario_motion2_2(self): # 요미가 주사 싫다고 우는 장면
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[220], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[150], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[95], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[9], motor_positions=[230], motor_speeds=[6])
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[5], motor_positions=[130], motor_speeds=[6])
        # rospy.sleep(2)
        # self.defalt_motion()
        # self.defalt_motion2()

    def scenario_motion2_3(self): # 요미가 두렵지만 주사를 맞는다고 하는 장면
        self.defalt_motion()
        self.defalt_motion2()
        rospy.sleep(1.5)
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[95], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[7])
        # rospy.sleep(3)
        # self.defalt_motion()
        # self.defalt_motion2()
        
    def scenario_switch1(self): # 머리 스위치
        self.defalt_motion()
        self.defalt_motion2()
        rospy.sleep(1.5)
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[95], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[265], motor_speeds=[7])
        
    def scenario_switch2(self): # 등 스위치
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[180, 180], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[180, 180], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[4, 8], motor_positions=[110, 250], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[180, 180], motor_speeds=[6])
        rospy.sleep(2)
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[190], motor_speeds=[8])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[230], motor_speeds=[7])
        rospy.sleep(0.2)
    
    def scenario_switch3_1(self): # 팔 스위치 왼쪽
        self.defalt_motion()
        self.defalt_motion2()
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[100], motor_speeds=[7])
        rospy.sleep(0.2)
    
    def scenario_switch3_2(self): # 팔 스위치 오른쪽
        self.defalt_motion()
        self.defalt_motion2()
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[260], motor_speeds=[6])
        rospy.sleep(0.2)

    def scenario_switch4_1(self): # 손 스위치 왼쪽
        self.defalt_motion()
        self.defalt_motion2()
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[7], motor_positions=[130], motor_speeds=[7])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
        rospy.sleep(1.5)
        for i in range(6):
            if i % 2 != 0:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[180], motor_speeds=[7])
                rospy.sleep(1)
            else:
                self.executor.motor_publisher_batch(motor_ids=[10], motor_positions=[230], motor_speeds=[7])
                rospy.sleep(1)

    def scenario_switch4_2(self): # 손 스위치 오른쪽
        self.defalt_motion()
        self.defalt_motion2()
        rospy.sleep(1)
        self.executor.motor_publisher_batch(motor_ids=[3], motor_positions=[230], motor_speeds=[6])
        rospy.sleep(0.2)
        self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[130], motor_speeds=[6])
        rospy.sleep(1.5)
        for i in range(6):
            if i % 2 != 0:
                self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[180], motor_speeds=[6])
                rospy.sleep(1)
            else:
                self.executor.motor_publisher_batch(motor_ids=[6], motor_positions=[130], motor_speeds=[6])
                rospy.sleep(1)

    def I_joy1(self):
        """(좋아하는 것에 대해) 팔을 살랑살랑 움직인다."""
        # self.executor.request_all_data()
        self.defalt_motion()
        self.frontal()
        self.stand()
        rospy.sleep(0.1)
        for i in range(3):
            self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[160, 160], motor_speeds=[5, 5])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[200, 200], motor_speeds=[5, 5])
            rospy.sleep(1.5)

    def I_joy2(self):
        "몸을 좌우로 조금씩 움직인다."
        self.defalt_motion()
        self.frontal()
        self.mover.rotate_in_place(30)
        rospy.sleep(0.2)
        self.mover.rotate_in_place(-30)
        rospy.sleep(0.2)

    def I_joy3(self):
        "가까운 사용자에게 다가간다."
        self.defalt_motion()
        self.frontal()
        obs = self.automover.get_obstacle_position(threshold=2.0)
        if obs is None:
            rospy.loginfo("대상이 감지 되지 않았습니다.")
            return
        obs_x, obs_y = obs
        self.automover.move_to_goal(obs_x, obs_y)

    def I_trust1(self):
        "신뢰하는 사용자 옆에서 따라다님"
        self.defalt_motion()
        self.frontal()
        self._vision_information()
        detect = self.vision_detect
        if detect == "people":
            obs = self.automover.get_obstacle_position(threshold=2.0)
            if obs is None:
                rospy.loginfo("대상이 감지 되지 않았습니다.")
                return
            obs_x, obs_y = obs

            target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "right")

            self.automover.move_to_goal(target_x, target_y)

    def I_trust2(self):
        "/사용자 지시가 있으면 즉각 응답 및 행동을 수행"

    def I_trust3(self):
        "/행동 및 이벤트 뒤 사용자 쪽으로 돌아옴"

    def I_fear1(self):
        "(위협 감지시)모서리로 움직인다 벽을 보고 선다"
        self.defalt_motion()
        self.defalt_motion2()
        self.automover.get_nearest_wall()
        self.automover.go_to_nearest_wall()
        
    def I_fear2(self):
        "신뢰하는 사용자 뒤로 숨는다"
        self.defalt_motion()
        self.frontal()
        self._vision_information()
        detect = self.vision_detect
        if detect == "people":
            obs = self.automover.get_obstacle_position(threshold=2.0)
            if obs is None:
                rospy.loginfo("대상이 감지 되지 않았습니다.")
                return
            obs_x, obs_y = obs

            target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "back")

            self.automober.move_to_goal(target_x, target_y)

    def I_fear3(self):
        "큰 물건 뒤로 숨는다"
        self.defalt_motion()
        self.frontal()
        obs = self.automover.get_obstacle_position(threshold=2.0)
        if obs is None:
            rospy.loginfo("대상이 감지 되지 않았습니다.")
            return
        obs_x, obs_y = obs

        target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "back")

        self.automober.move_to_goal(target_x, target_y)

    def I_surprise1(self):
        "/(소리-(어? 뭐지?) 작은 목소리 출력 후) 바퀴가 뒤로 살짝 밀리며 정지"
        self.mover.move_backward(1.5)
        rospy.sleep(0.2)
        self.stand()

    def I_surprise2(self):
        "미세 움직임으로 진동 표현, 떨림 표현"
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[6])
        rospy.sleep(0.2)
        for i in range(3):
            self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[90, 270], motor_speeds=[5, 5])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[200, 160], motor_speeds=[5, 5])
            rospy.sleep(1.5)

    def I_surprise3(self):
        "/((호불호에서 호의 경우) 대상으로부터 가까워지고, (불호의 경우) 대상으로부터 멀어짐)"

    def I_sadness1(self):
        "느리게 이동"
        self.defalt_motion()
        self.mover.move_forward(1.0)

    def I_sadness2(self):
        "고개를 숙이듯 머리가 아래로 향함"
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[159], motor_speeds=[5])
        rospy.sleep(1.5)

    def I_sadness3(self):
        "일정 시간 정지 상태 유지"
        self.defalt_motion()
        self.stand()
        rospy.sleep(1.5)

    def I_disgust1(self):
        "/원치 않는 사용자 접근 시 후진"

    def I_disgust2(self):
        "회피 루트를 탐색"
        self.defalt_motion()
        self.wait_command()
        self.mover.move_backward(1.5)
    
    def I_disgust3(self):
        "도망, 회피, 뒤 돌기"
        self.defalt_motion()
        self.mover.rotate_in_place(180)
        rospy.sleep(2.0)
        self.mover.move_forward(1.5)

    def I_anger1(self):
        "본체의 바퀴가 빠르게 앞뒤로 움직임"
        self.defalt_motion()
        self.defalt_motion2()
        self.mover.move_forward(1.5)
        rospy.sleep(2.0)
        self.mover.move_backward(1.5)
        rospy.sleep(2.0)

    def I_anger2(self):
        "몸 전체에 진동(가능하면)"
        self.defalt_motion()
        for i in range(5):
            self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[98, 262], motor_speeds=[7])
            rospy.sleep(1.5)

    def I_anticipation1(self):
        "주변을 돌아다닌다"
        self.defalt_motion()
        self.frontal()

        side_length = 0.5
        angle = 90

        for i in range(4):
            self.mover.move_forward(side_length)
            rospy.sleep(0.5)
            self.mover.rotate_in_place(angle)
            rospy.sleep(0.5)

    def I_anticipation2(self):
        "바퀴로 자리에서 빙글빙글 천천히 돈다"
        self.defalt_motion()
        for i in range(6):
            self.mover.rotate_in_place(180)
    
    def I_anticipation3(self):
        "새로운 입력을 기다리는 듯 정지 후 전방 주시" # 팔 내리는거 추가
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[220], motor_speeds=[5])

    def E_joy1(self):
        "/(행위가 끝난뒤) 크게 공간을 한바퀴 돌며 정해진 위치로 돌아감"
        self.defalt_motion()
        self.frontal()

        side_length = 0.5
        angle = 90

        for i in range(4):
            self.mover.move_forward(side_length)
            rospy.sleep(0.5)
            self.mover.rotate_in_place(angle)
            rospy.sleep(0.5)
    
    def E_joy2(self):
        "정위치에서 양손을 든다."
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[3, 7], motor_positions=[300, 60], motor_speeds=[7])
        rospy.sleep(1.5)

    def E_joy3(self):
        "바퀴로 신나게 빙빙 돈다"
        self.defalt_motion()
        for i in range(10):
            self.mover.rotate_in_place(180)

    def E_trust1(self):
        "/신뢰대상에게 일정거리 유지하며 옆에 선다."
        self.defalt_motion()
        self.frontal()
        self._vision_information()
        detect = self.vision_detect
        if detect == "people":
            obs = self.automover.get_obstacle_position(threshold=2.0)
            if obs is None:
                rospy.loginfo("대상이 감지 되지 않았습니다.")
                return
            obs_x, obs_y = obs

            target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "right")
            self.automober.move_to_goal(target_x, target_y)
    
    def E_trust2(self):
        "일정거리 유지하며 따라다닌다."
        self.defalt_motion()
        self.frontal()
        self._vision_information()
        detect = self.vision_detect
        while detect == "people":
            obs = self.automover.get_obstacle_position(threshold=2.0)
            if obs is None:
                rospy.loginfo("대상이 감지 되지 않았습니다.")
                return
            obs_x, obs_y = obs

            target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "back")

            self.automober.move_to_goal(target_x, target_y)

    def E_trust3(self):
        "/사용자 지시가 있으면 즉각 응답 및 행동을 수행하고 대기한다."
    
    def E_fear1(self):
        "(위협감지 시)모서리로 움직인다 벽을 보고 선다."
        self.defalt_motion()
        self.defalt_motion2()
        self.automover.get_nearest_wall()
        self.automover.go_to_nearest_wall()
    
    def E_fear2(self):
        "신뢰하는 사용자 뒤로 숨는다."
        self.defalt_motion()
        self.frontal()
        self._vision_information()
        detect = self.vision_detect
        if detect == "people":
            obs = self.automover.get_obstacle_position(threshold=2.0)
            if obs is None:
                rospy.loginfo("대상이 감지 되지 않았습니다.")
                return
            obs_x, obs_y = obs

            target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "back")

            self.automober.move_to_goal(target_x, target_y)
    
    def E_fear3(self):
        "큰 물건 뒤로 숨는다"
        self.defalt_motion()
        self.frontal()
        obs = self.automover.get_obstacle_position(threshold=2.0)
        if obs is None:
            rospy.loginfo("대상이 감지 되지 않았습니다.")
            return
        obs_x, obs_y = obs

        target_x, target_y = self.automover.get_relative_position(obs_x, obs_y, offset = 0.5, direction = "back")

        self.automover.move_to_goal(target_x, target_y)

    def E_surprise1(self):
        "/(소리-(어? 뭐지?) 작은 목소리 출력 후) 바퀴가 뒤로 살짝 밀리며 정지"
        self.mover.move_backward(1.5)
        rospy.sleep(0.2)
        self.stand()

    def E_surprise2(self):
        "미세 움직임으로 진동 표현, 떨림 표현"
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[6])
        rospy.sleep(0.2)
        for i in range(3):
            self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[90, 270], motor_speeds=[5, 5])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[5, 9], motor_positions=[200, 160], motor_speeds=[5, 5])
            rospy.sleep(1.5)

    def E_surprise3(self):
        "/((호불호에서 호의 경우) 대상으로부터 가까워지고, (불호의 경우) 대상으로부터 멀어짐)"

    def E_sadness1(self):
        "느리게 이동"
        self.defalt_motion()
        self.mover.move_forward(1.0)

    def E_sadness2(self):
        "고개를 숙이듯 머리가 아래로 향함"
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[159], motor_speeds=[5])
        rospy.sleep(1.5)

    def E_sadness3(self):
        "벽 쪽에 멈춰서 혼자 조용히 있기"

    def E_disgust1(self):
        "원치 않은 사용자 접근 시 후진"
        self.defalt_motion()
        self._vision_information()
        detect = self.vision_detect
        if detect == "people":
            self.mover.move_backward(1.5)
            rospy.sleep(0.2)
            self.stand()

    def E_disgust2(self):
        "옆으로 돌아서기(고개를 돌리는 것과 같은 효과 일거 같아서 그렇게 만듬)"
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[12], motor_positions=[240], motor_speeds=[5])
        rospy.sleep(1.5)

    def E_disgust3(self):
        "도망가기"
        self.defalt_motion()
        self.mover.move_backward(3.0)
        rospy.sleep(0.2)
    
    def E_anger1(self):
        "본체의 바퀴가 빠르게 앞뒤로 움직임"
        self.defalt_motion()
        self.mover.move_forward(1.5)
        rospy.sleep(2.0)
        self.mover.move_backward(1.5)
        rospy.sleep(2.0)

    def E_anger2(self):
        "몸 전체에 진동(가능하면)"
        self.defalt_motion()
        for i in range(5):
            self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[95, 265], motor_speeds=[7])
            rospy.sleep(1.5)
            self.executor.motor_publisher_batch(motor_ids=[6, 10], motor_positions=[98, 262], motor_speeds=[7])
            rospy.sleep(1.5)

    def E_anger3(self):
        "주변을 빠르게 돌아다닌다"
        self.defalt_motion()
        self.frontal()

        self.mover.speed = 0.5
        side_length = 0.5
        angle = 90

        for i in range(4):
            self.mover.move_forward(side_length)
            rospy.sleep(0.5)
            self.mover.rotate_in_place(angle)
            rospy.sleep(0.5)

    def E_anticipation1(self):
        "주변을 돌아다닌다"
        self.defalt_motion()
        self.frontal()

        side_length = 0.5
        angle = 90

        for i in range(4):
            self.mover.move_forward(side_length)
            rospy.sleep(0.5)
            self.mover.rotate_in_place(angle)
            rospy.sleep(0.5)

    def E_anticipation2(self):
        "바퀴로 자리에서 빙글빙글 천천히 돈다"
        self.defalt_motion()
        for i in range(6):
            self.mover.rotate_in_place(180)
    
    def E_anticipation3(self):
        "새로운 입력을 기다리는 듯 정지 후 전방 주시"
        self.defalt_motion()
        self.executor.motor_publisher_batch(motor_ids=[11], motor_positions=[220], motor_speeds=[6])
        rospy.sleep(1.5)

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
    def run_motion(self, func_name: str):
        if hasattr(self, func_name):
            func = getattr(self, func_name)
            if callable(func):
                try:
                    func()
                except Exception as e:
                    print(f"[MotionController] '{func_name}' 실행 오류: {e}")
            else:
                print(f"[MotionController] '{func_name}'은 호출 불가능한 항목입니다.")
        else:
            print(f"[MotionController] 해당 함수 '{func_name}' 존재하지 않음")

if __name__ == '__main__':
    # 터미널 직접 입력
     try:
         controller = MotionController()
         while not rospy.is_shutdown():
             func_name = input("함수이름(종료는 end) ")
             if func_name.lower() == 'end':
                 break
             elif hasattr(controller, func_name):
                 func = getattr(controller, func_name)
                 if callable(func):
                     try:
                         func()
                     except Exception as e:
                         print(f"함수 실행 중 오류 발생: {e}")
                 else:
                     print("호출 가능한 함수가 아님.")
             else:
                 print("해당하는 함수 x")
     except rospy.ROSInterruptException:
         pass
    #try:
    #   controller = MotionController()
    #    while not rospy.is_shutdown():
    #        func_name = input("함수이름(종료는 end) ")
    #        if func_name.lower() == 'end':
    #            break
    #        controller.run_motion(func_name)
    #except rospy.ROSInterruptException:
    #    pass