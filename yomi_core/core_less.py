import sys 
import os 
# BDIA.py가 있는 폴더: .../4IU_ROBOTAI/yomi_core 
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 프로젝트 루트: .../4IU_ROBOTAI 
PROJECT_ROOT = os.path.dirname(THIS_DIR) 

# PYTHONPATH에 루트와 서브패키지 추가 
sys.path.insert(0, PROJECT_ROOT) 
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yomi_core")) 
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yomi_motor")) 
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yomi_core", "tts")) 
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yomi_core", "vision_face")) 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import threading 
import random 
import time 
import re 
import json 
from sensor_msgs.msg import Joy #joystick 값 

#ros통신용 
import rospy, rosgraph 
from std_msgs.msg import String, Bool 
from yomi_motor.scripts.yomi_motion import MotionController 
from yomi_core.tts.TTS_server import TTSServer, VITS, TTSClient 
from yomi_core.vision_face.VisionFace import VisonFaceMain 
#5번 임시로 시도 중 
from std_msgs.msg import String 
from sensor_msgs.msg import Joy 

class BDIA: 
    def __init__(self): 
        # __init__ 내부 초반
        self.vits_model = None
        self.tts_server = None
        self.tts = None

        # VITS 서버 기동
        try:
            self.vits_model = VITS()
            self.tts_server = TTSServer(self.vits_model)
            threading.Thread(target=self.tts_server.run, daemon=True).start()
            rospy.loginfo("[YOMI] TTSServer : 준비완료")
        except Exception as e:
            rospy.logwarn(f"[YOMI] TTSServer 초기화 실패: {e}")

        # TTS 클라이언트 준비 (콜백은 방금 만든 메서드 연결)
        try:
            self.tts = TTSClient()
            rospy.loginfo("[YOMI] TTSClient : 준비완료")
        except Exception as e:
            rospy.logwarn(f"[YOMI] TTSClient 초기화 실패: {e}")
            self.tts = None


    # 1) roscore 확인 
        try: 
            import rosgraph 
            if not rosgraph.is_master_online(): 
                raise RuntimeError("ROS master(roscore)가 실행 중이 아닙니다.") 
        except Exception as e: 
            print(f"[BDIA] 경고: {e}") 

    # 2) 노드 초기화 (다른 곳에서 init_node를 또 호출하지 않도록 주의) 
        if not rospy.core.is_initialized(): 
            rospy.init_node('BDIA', anonymous=True) 
            
    # 3) 모터 컨트롤러 준비 (예외 대비) 
        self.motor_controller = None 
        try: 
            self.motor_controller = MotionController() 
            rospy.loginfo("MotionController initialized.") 
        except Exception as e: rospy.logwarn(f"MotionController 초기화 실패: {e}") 

    # # 4) 조이스틱 엣지 검출용 상태 
    #     self.prev_buttons = None

    # 5) 콜백 등록
        self.last_press_time = {}
        self.last_press_time_switch = None
        rospy.Subscriber('/joy', Joy, self.joy_callback) 
        rospy.on_shutdown(self._on_shutdown)
        
    # # 6) 최초 연결 확인 (없어도 동작엔 지장 없음) 
    #     try: 
    #         msg = rospy.wait_for_message('/joy', Joy, timeout=5.0) 
    #         rospy.loginfo(f"/joy 연결 확인: buttons={tuple(msg.buttons)}") 
    #         # 첫 수신 값을 이전 상태로 저장해 ‘꾹 누름’ 반복 호출 방지 
    #         self.prev_buttons = list(msg.buttons) 
    #     except rospy.ROSException: 
    #         rospy.logwarn("5초 내 /joy 수신 없음. joy_node 실행/장치/네임스페이스 확인 필요.") 
    #         # 최초 미수신 시에도 콜백에서 안전하게 동작하도록 None 유지 

        #ROS Subscriber
        rospy.Subscriber('/switch_1_state', Bool, self.handle_switch, callback_args=1)
        rospy.Subscriber('/switch_2_state', Bool, self.handle_switch, callback_args=2)
        rospy.Subscriber('/switch_3_state', Bool, self.handle_switch, callback_args=3)
        rospy.Subscriber('/switch_4_state', Bool, self.handle_switch, callback_args=4)
        rospy.Subscriber('/switch_5_state', Bool, self.handle_switch, callback_args=5)
        rospy.Subscriber('/switch_6_state', Bool, self.handle_switch, callback_args=6)
        print("[YOMI] ROS Subscriber: 토픽(switch_state) 연결완료")

        self.emotions = ["joy", "sadness", "angry", "fear", "surprise", "disgust", "trust", "anticipation", "no"]
        self.emotion_index = 0
        self.VisionFace = VisonFaceMain(interval=2, viewGUI=True, mbti="I")
        self.VisionFace.run()

    def handle_switch(self, msg, index): 
        """
        ROS토픽 이용 Switch 처리함수
        1: "등",
        2: "왼팔",
        3: "오른팔",
        4: "왼손",
        5: "오른손",
        6: "머리"
        """
        print(f"index:{index}")
        now = time.time() 
        state = True if msg.data else False
        if state: 
            if self.last_press_time_switch is None or (now - self.last_press_time_switch) > 0.5:
                self.last_press_time_switch = now
                if index == 1:
                    self.motion_switch1()
                    rospy.loginfo("switch1 push")
        
                elif index == 2:
                    self.motion_switch2()
                    rospy.loginfo("switch2 push")

                elif index == 4:
                    self.motion_switch4()
                    rospy.loginfo("switch4 push")
                

                elif index == 5:
                    self.motion_switch5()
                    rospy.loginfo("switch5 push")
                

                elif index == 6:
                    self.motion_switch6()
                    rospy.loginfo("switch6 push")
            
        
    def joy_callback(self, msg): 
        """ f710버튼 pressed, released를 구분하는 joystick콜백함수 stick값 확장가능 """ 
        now = time.time() 
        if msg.axes[0] > 0.9: 
            for i in range(len(msg.buttons)): 
                if msg.buttons[i] == 1: 
                    if i not in self.last_press_time or (now - self.last_press_time[i]) > 0.5: 
                        self.last_press_time[i] = now 
                        if i == 3: # Y 
                            self.motion_leftY() 
                            rospy.loginfo("left + Y pressed") 
                        elif i == 2: # X 
                            self.motion_leftX() 
                            rospy.loginfo("left + X pressed") 
                        return
                    
        elif msg.axes[0] < -0.9:
            for i in range(len(msg.buttons)): 
                if msg.buttons[i] == 1: 
                    if i not in self.last_press_time or (now - self.last_press_time[i]) > 0.5: 
                        self.last_press_time[i] = now 
                        if i == 3: # Y 
                            self.motion_rightY() 
                            rospy.loginfo("right + Y pressed") 
                        elif i == 2: # X 
                            self.motion_rightX() 
                            rospy.loginfo("right + X pressed") 
                        elif i == 1: # B 
                            self.motion_rightB() 
                            rospy.loginfo("right + B pressed")

        elif msg.axes[1] > 0.9:
            for i in range(len(msg.buttons)): 
                if msg.buttons[i] == 1: 
                    if i not in self.last_press_time or (now - self.last_press_time[i]) > 0.5: 
                        self.last_press_time[i] = now 
                        if i == 3: # Y 
                            self.motion_upY() 
                            rospy.loginfo("up + Y pressed") 
                        elif i == 2: # X 
                            self.motion_upX() 
                            rospy.loginfo("up + X pressed") 
                        elif i == 1: # B 
                            self.motion_upB() 
                            rospy.loginfo("up + B pressed")

        elif msg.axes[1] < -0.9:
            for i in range(len(msg.buttons)): 
                if msg.buttons[i] == 1: 
                    if i not in self.last_press_time or (now - self.last_press_time[i]) > 0.5: 
                        self.last_press_time[i] = now 
                        if i == 3: # Y 
                            self.motion_downY() 
                            rospy.loginfo("down + Y pressed") 
                        elif i == 2: # X 
                            self.motion_downX() 
                            rospy.loginfo("down + X pressed") 
        
        for i in range(len(msg.buttons)): 
            if msg.buttons[i] == 1: 
                if i not in self.last_press_time or (now - self.last_press_time[i]) > 0.5: 
                    self.last_press_time[i] = now 
                    # for i in [5, 4, 3, 2, 1, 0]: # A (0), B (1), X(2), Y(3) 
                        #self.last_press_time[i] = now
                    if i == 2: # X 
                        self.motionX() 
                        rospy.loginfo("X pressed") 
                    elif i == 1: # B 
                        self.motionB() 
                        rospy.loginfo("B pressed") 
                    elif i == 0: # A 
                        self.motionA() 
                        rospy.loginfo("A pressed") 
                    elif i == 5: # RB 
                        self.emotion_index = (self.emotion_index + 1) % len(self.emotions) 
                        self.VisionFace.face_set_emotion(self.emotions[self.emotion_index])
                    # elif i == 4: # LB 
                    #     self.emotion_index -= (self.emotion_index - 1) % len(self.emotions)
                    #     self.VisionFace.face_set_emotion(self.emotions[self.emotion_index]) 
                    elif i == 6: #start 
                        self.emotion_index = 7 
                        self.VisionFace.face_set_emotion(self.emotions[self.emotion_index]) 
                    elif i == 7: #back 
                        self.emotion_index = 2 
                        self.VisionFace.face_set_emotion(self.emotions[self.emotion_index])
                     
    def motionX(self):
        """차렷!!"""
        self.VisionFace.face_set_emotion("no")
        threading.Thread(target=self._motionX_sequence, daemon=True).start()
    
    def _motionX_sequence(self):
        self.motor_controller.filiming_motion2()

    def motionB(self): 
        """시나리오 1-3"""
        threading.Thread(target=self._motionB_sequence, daemon=True).start()
        self.VisionFace.face_set_emotion("joy") 
        try:
            if self.tts:
                self.tts.send_text("나는 오늘 현빈이랑 공놀이 했어!") #기쁜표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def _motionB_sequence(self):
        self.motor_controller.homshowping1()

        
    def motionA(self): 
        """시나리오 1-4"""
        threading.Thread(target=self._motionX_sequence, daemon=True).start()
        self.VisionFace.face_set_emotion("angry")
        try:
            if self.tts:
                self.tts.send_text("안돼! 거짓말은 나쁜거야") #화난 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
        

    def motion_leftY(self): 
        """시나리오 2"""
        self.VisionFace.face_set_emotion("joy")
        try:
            if self.tts:
                self.tts.send_text("안녕! 나는 요미야.") # 기쁜표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
        threading.Thread(target=self._motionleftY_sequence, daemon=True).start()

        threading.Thread(target=self._motionleftY2_sequence, daemon=True).start()
        try:
            if self.tts:
                self.tts.send_text("너는 누구야?") 
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")

    def _motionleftY_sequence(self):
        self.motor_controller.Hi()

    def _motionleftY2_sequence(self):
        rospy.sleep(7)
        self.motor_controller.Hi2()
           
    def motion_leftX(self): 
        """시나리오 3"""
        self.VisionFace.face_set_emotion("angry")
        try:
            if self.tts:
                self.tts.send_text("아야 이건 좀 아픈데") #화난 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
        threading.Thread(target=self._motionleftX_sequence, daemon=True).start()

    def _motionleftX_sequence(self):
        self.motor_controller.scenario_motion2_3()

    def motion_rightY(self): 
        """시나리오 1-2""" 
        threading.Thread(target=self._motionrightY_sequence, daemon=True).start()
        self.VisionFace.face_set_emotion("joy")
        try:
            if self.tts:
                self.tts.send_text("야호 맜있는거 좋아!") 
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")

    def _motionrightY_sequence(self):
        self.motor_controller.scenario_motion2_1()

    def motion_rightX(self):
        threading.Thread(target=self._motionrightX_sequence, daemon=True).start() 
        self.VisionFace.face_set_emotion("sadness")
        try:
            if self.tts:
                self.tts.send_text("안돼.. 주사는 아파서 싫어") 
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")

    def _motionrightX_sequence(self):
        self.motor_controller.scenario_motion2_2()

    def motion_rightB(self):
        threading.Thread(target=self._motionrightB_sequence, daemon=True).start()
        self.VisionFace.face_set_emotion("fear")
        try:
            if self.tts:
                self.tts.send_text("알겠어... 난 용감하니까 주사 한 번 맞아볼게") 
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")

    def _motionrightB_sequence(self):
        self.motor_controller.scenario_motion2_3()

    def motion_switch1(self):
        self.VisionFace.face_set_emotion("no")
        threading.Thread(target=self.motion_switch1_sequence_test, daemon=True).start()
        threading.Thread(target=self.motion_switch1_sequence, daemon=True).start()
    
    def motion_switch1_sequence(self):
        self.motor_controller.scenario_switch2()

    def motion_switch1_sequence_test(self):
        try:
            if self.tts:
                self.tts.send_text("누구세요") # 무표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def motion_switch2(self):
        self.VisionFace.face_set_emotion("angry")
        threading.Thread(target=self.motion_switch2_sequence, daemon=True).start()
        try:
            if self.tts:
                self.tts.send_text("하지마세요") # 화난 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def motion_switch2_sequence(self):
        self.motor_controller.scenario_switch3_1()
        rospy.sleep(2)

    def motion_switch3(self):
        self.VisionFace.face_set_emotion("angry")
        threading.Thread(target=self.motion_switch3_sequence, daemon=True).start()
        try:
            if self.tts:
                self.tts.send_text("하지마세요") # 화난 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def motion_switch3_sequence(self):
        self.motor_controller.scenario_switch3_2()
        rospy.sleep(2)

    def motion_switch4(self):
        self.VisionFace.face_set_emotion("joy")
        threading.Thread(target=self.motion_switch4_sequence, daemon=True).start()
        try:
            if self.tts:
                self.tts.send_text("악수하자") # 기쁜 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def motion_switch4_sequence(self):
        self.motor_controller.scenario_switch4_1()
        rospy.sleep(2)

    def motion_switch5(self):
        self.VisionFace.face_set_emotion("joy")
        threading.Thread(target=self.motion_switch5_sequence, daemon=True).start()
        try:
            if self.tts:
                self.tts.send_text("악수하자") # 기쁜 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def motion_switch5_sequence(self):
        self.motor_controller.scenario_switch4_2()
        rospy.sleep(2)

    def motion_switch6(self):
        self.VisionFace.face_set_emotion("angry")
        threading.Thread(target=self.motion_switch6_sequence, daemon=True).start()
        try:
            if self.tts:
                self.tts.send_text("아야 이건 좀 아픈데") # 기쁜 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")
    
    def motion_switch6_sequence(self):
        self.motor_controller.scenario_switch1()
        rospy.sleep(2)

    def motion_upY(self):
        self.VisionFace.face_set_emotion("anticipation")
        threading.Thread(target=self.motion_upY_sequence, daemon=True).start()

    def motion_upY_sequence(self):
        self.motor_controller.gababo_ba()

    def motion_upX(self):
        self.VisionFace.face_set_emotion("anticipation")
        threading.Thread(target=self.motion_upX_sequence, daemon=True).start()

    def motion_upX_sequence(self):
        self.motor_controller.gababo_ga()

    def motion_upB(self):
        self.VisionFace.face_set_emotion("anticipation")
        threading.Thread(target=self.motion_upB_sequence, daemon=True).start()

    def motion_upB_sequence(self):
        self.motor_controller.gababo_bo()

    def motion_downY(self):
        self.VisionFace.face_set_emotion("joy")
        try:
            if self.tts:
                self.tts.send_text("야호 이겼다.") # 기쁜 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")

    def motion_downX(self):
        self.VisionFace.face_set_emotion("sadness")
        try:
            if self.tts:
                self.tts.send_text("힝 졌어") # 기쁜 표정
            else:
                rospy.logwarn("[TTS] Client 미준비")
        except Exception as e:
            rospy.logwarn(f"[TTS] 발화 실패: {e}")

    def _on_shutdown(self): 
        try: 
            self.stop() 
        except Exception as e: 
            rospy.logwarn(f"BDIA stop during on_shutdown: {e}") 
            
    def stop(self): 
        rospy.loginfo("BDIA stopped.") 
        
if __name__ == "__main__": 
    bdia = BDIA() 
    # bdia.motor_controller = MotionController() 

    try: 
        rospy.loginfo("BDIA node running. Listening to /joy ...") 
        rospy.spin() 

    except KeyboardInterrupt: 
        rospy.loginfo("KeyboardInterrupt detected. Shutting down...") 
        rospy.signal_shutdown("KeyboardInterrupt") 
        
    finally: 
        bdia.stop()