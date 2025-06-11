import sys
import os
import time
import rospy
import threading
from std_msgs.msg import String, Bool

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
        rospy.Subscriber('/llm_emotion', String, self.change_face)
        rospy.Subscriber('/tts_state', Bool, self.tts_face)

        #module init
        self.yomi_core = Yomi(isSTT=True, isLLM=False, isTTS=True, isVision=False)
        self.yomi_face = Yomi_face()
        # self.yomi_face = None
    
    def start(self):
        threading.Thread(target=self.yomi_core.start, daemon=True).start()
        self.yomi_face.start()
    
    def stop(self):
        self.yomi_core.stop()
        self.yomi_face.stop()
    
    def change_face(self, msg):
        emotion = msg.data.strip()
        self.yomi_face.change_face(emotion)
    
    def tts_face(self, msg):
        is_tts = msg.data
        self.yomi_face.toggle_emotion(is_tts)

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
