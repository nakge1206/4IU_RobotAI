import sys
import os
import time
import rospy
import threading
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Joy #joystick 값

sys.path.append(os.path.join(os.path.dirname(__file__), 'yomi_core/'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'yomi_face/'))
from yomi_core.core import Yomi
from yomi_face.kao import Yomi_face

class Gwiyomi:
    def __init__(self):
        #ros Node init
        if not rospy.core.is_initialized():
            rospy.init_node('gwiyomi', anonymous=True)
        
        #ros subscriber
        # rospy.Subscriber('/llm_emotion', String, self.handle_emotion)
        # rospy.Subscriber('/tts_state', Bool, self.tts_face)
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        #Joystick
        self.prev_buttons = [0] * 10 #Joystick의 버튼의 상태변화를 저장하기 위한 변수

        #module init
        self.yomi_core = Yomi(isSTT=True, isLLM=False, isTTS=True, isVision=False)
        # self.yomi_face = Yomi_face()
        self.emotion_map = {
            "기쁨": "joy",
            "슬픔": "sadness",
            "분노": "angry",
            "공포": "fear",
            "놀라움": "surprise",
            "혐오": "disgust",
            "신뢰": "trust",
            "기대": "anticipation"
        }
    
    def start(self):
        self.yomi_core.start()
        # self.yomi_face.start()
    
    def stop(self):
        self.yomi_core.stop()
        # self.yomi_face.stop()

    def joy_callback(self, msg):
        """
        f710버튼 pressed, released를 구분하는 joystick콜백함수
        stick값 확장가능
        """
        for i in [3, 1]:  # Y (3), B (1)
            if msg.buttons[i] == 1 and self.prev_buttons[i] == 0:
                if i == 3:
                    self.yomi_core.resume()
                    print("Y pressed")
                    # rospy.loginfo("Y pressed")
                elif i == 1:
                    self.yomi_core.pause()
                    print("B pressed")
                    # rospy.loginfo("B pressed")
            if msg.buttons[i] == 0 and self.prev_buttons[i] == 1:
                if i == 3:
                    rospy.loginfo("Y released")
                elif i == 1:
                    rospy.loginfo("B released")
        self.prev_buttons = list(msg.buttons)

    # def handle_emotion(self, msg):
    #     kor_emotion = msg.data.strip()
    #     eng_emotion = self.emotion_map.get(kor_emotion, "joy")  # 기본값은 joy
    #     if self.yomi_face:
    #         self.yomi_face.command_queue.put(("change_face", (eng_emotion,)))
    #     # self.yomi_face.change_face(eng_emotion)
    
    # def tts_face(self, msg):
    #     is_tts = msg.data
    #     print(str(is_tts))
    #     if self.yomi_face:
    #         self.yomi_face.command_queue.put(("toggle_emotion", (is_tts,)))
    #     # self.yomi_face.toggle_emotion(is_tts)

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
