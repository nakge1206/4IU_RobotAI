import sys
import os
import time
import rospy
import threading
from std_msgs.msg import String

import queue

#모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'yomi_face/'))
from yomi_face.kao import Yomi_face

class Gwiyomi:
    def __init__(self):
        #ros Node init
        if not rospy.core.is_initialized():
            rospy.init_node('gwiyomi', anonymous=True)
        
        #ros subscriber
        # rospy.Subscriber('/llm_emotion', String, self.handle_emotion)

        #ros publisher
        self.llm_emotion = rospy.Publisher('/play_motion_sequence', String, queue_size=10)

        #module init
        self.yomi_face = Yomi_face()
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
        self.emotion_list = ["기쁨", "슬픔", "분노", "공포", "놀라움", "혐오", "신뢰", "기대"]
    
    def start(self):
        threading.Thread(target=self.run_emotions_loop, daemon=True).start()
        self.yomi_face.start()
    
    def stop(self):
        self.yomi_face.stop()

    def run_emotions_loop(self):
        for kor_emotion in self.emotion_list:
            if not rospy.is_shutdown():
                eng_emotion = self.emotion_map.get(kor_emotion, "joy")
                rospy.loginfo(f"Setting emotion: {kor_emotion} -> {eng_emotion}")
                self.llm_emotion.publish(eng_emotion)
                self.yomi_face.blinking=True
                self.yomi_face.set_emotion(eng_emotion)
                time.sleep(10)

    # #yomi_face 관련함수
    # def handle_emotion(self, msg):
    #     kor_emotion = msg.data.strip()
    #     eng_emotion = self.emotion_map.get(kor_emotion, "joy")  # 기본값은 joy
    #     self.llm_emotion.publish(eng_emotion)
    #     self.yomi_face.set_emotion(eng_emotion)

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

    # try:
    #    service.run_emotions_loop()

    # except KeyboardInterrupt:
    #     print("\n 종료 중...")
    # finally:
    #     service.stop()
