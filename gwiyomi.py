import sys
import os
import time
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Joy #joystick 값

import queue

#모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'yomi_core/'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'yomi_face/'))
from yomi_core.core import Yomi
# from yomi_face.kao import Yomi_face

class Gwiyomi:
    def __init__(self):
        #ros Node init
        if not rospy.core.is_initialized():
            rospy.init_node('gwiyomi', anonymous=True)
        
        #ros subscriber
        # rospy.Subscriber('/llm_emotion', String, self.handle_emotion)
        # rospy.Subscriber('/tts_state', Bool, self.tts_face)
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        #ros publisher
        self.llm_emotion = rospy.Publisher('/play_motion_sequence', String, queue_size=10)

        #Joystick
        # self.prev_buttons = [0] * 10 #Joystick의 버튼의 상태변화를 저장하기 위한 변수
        self.is_joy_y = False
        self.is_joy_b = False

        #module init
        self.yomi_core = Yomi(isSTT=True, isTTS=False, isLLM=False, isVisionFace=True)
        # self.yomi_face = Yomi_face()
        # self.emotion_map = {
        #     "기쁨": "joy",
        #     "슬픔": "sadness",
        #     "분노": "angry",
        #     "공포": "fear",
        #     "놀라움": "surprise",
        #     "혐오": "disgust",
        #     "신뢰": "trust",
        #     "기대": "anticipation"
        # }
    
    def start(self):
        self.yomi_core.start()
        # self.yomi_face.start()
    
    def stop(self):
        self.yomi_core.stop()
        # self.yomi_face.stop()
    
    def core_pause(self):
        self.yomi_core.joy_master_flag = False
    def core_resume(self):
        self.yomi_core.joy_master_flag = True

    def joy_callback(self, msg):
        """
        f710버튼 pressed, released를 구분하는 joystick콜백함수
        stick값 확장가능
        """
        for i in [3, 1]:  # Y (3), B (1)
            if msg.buttons[i] == 1:
                if i == 3: # Y
                    self.isJoyY(True)
                    rospy.loginfo("Y pressed")
                elif i == 1: # B
                    self.isJoyB(True)
                    rospy.loginfo("B pressed")
            
    def isJoyY(self, is_joyY):
        if(self.is_joy_y == True):
            if(is_joyY == True): # ON->ON
                pass
            else: #ON->OFF
                # rospy.loginfo("Y : ON->OFF")
                self.is_joy_y = False
        else:
            if(is_joyY == True): #OFF->ON
                # rospy.loginfo("Y : OFF->ON")
                self.is_joy_y = True
                self.isJoyB(False)
                self.core_resume()
            else: #OFF->OFF
                pass
    
    def isJoyB(self, is_joyB):
        if(self.is_joy_b == True):
            if(is_joyB == True): # ON->ON
                pass
            else: #ON->OFF
                # rospy.loginfo("B : ON->OFF")
                self.is_joy_b = False
        else:
            if(is_joyB == True): #OFF->ON
                # rospy.loginfo("B : OFF->ON")
                self.is_joy_b = True
                self.isJoyY(False)
                self.core_pause()
            else: #OFF->OFF
                pass

    #yomi_face 관련함수
    # def handle_emotion(self, msg):
    #     kor_emotion = msg.data.strip()
    #     eng_emotion = self.emotion_map.get(kor_emotion, "joy")  # 기본값은 joy
    #     self.llm_emotion.publish(eng_emotion)
    #     self.yomi_face.set_emotion(eng_emotion)
    
    # def tts_face(self, msg):
    #     is_tts = msg.data
    #     self.yomi_face.set_blinking(is_tts)

if __name__ == "__main__":
    service = Gwiyomi()
    service.start()

    try:
        while not rospy.is_shutdown():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n 종료 중...")
    finally:
        service.stop()
