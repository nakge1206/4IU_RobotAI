import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import threading
import random
import time
#ros통신용
import rospy
from std_msgs.msg import String, Bool

sys.path.append(os.path.join(os.path.dirname(__file__), 'vision_face'))

from vision_face.VisionFace import VisonFaceMain #Vision

class visiontest:
    def __init__(self):
        self.VisionFace = VisonFaceMain(interval=2, viewGUI=True)

    def run(self):
        self.VisionFace.run()

    def stop(self):
        self.stop()

if __name__ == "__main__":
    service = visiontest()
    service.run()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n 종료 중...")
    finally:
        service.stop()