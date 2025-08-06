import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import threading
import random
import time

import asyncio

#ros통신용
import rospy, rosgraph
from std_msgs.msg import String, Bool


# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'stt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision_face'))
sys.path.append(os.path.abspath('./yomi_motor/scripts'))


# 각 모듈 임포트
from stt.realtime_stt_module import STTModule           #STT
from tts.TTS_server import TTSServer, VITS, TTSClient   #TTS
from vision_face.VisionFace import VisonFaceMain        #Vision
from llm_core.inference_client_ax4 import LLMClient     #LLM
# from yomi_motor import MotionSequenceExecutor           #Motor


class Yomi:
    def __init__(self, isSTT=True, isTTS=True, isLLM=True, isVisionFace=True, mbti="I"):
        self.isSTT = isSTT
        self.isTTS = isTTS
        self.isLLM = isLLM
        self.isVisionFace = isVisionFace
        self.mbti = mbti

        #STT Timeout
        self.stt_timeout = 20 #stt_timeout-몇 초마다 Vision으로 LLM실행하는지
        self.sttTimer = None
        self.visionEnable = False

        #STT 활성화 조건
        self.sttEnable = True

        #TTS 작동 여부
        self.is_tts_running = False
        
        #각종 플래그
        self.joy_master_flag = True
        self.switch_flag = False
        self.vision_flag = False

        #시각정보 저장용
        self.lastVision = None

        #모듈 초기화
        self.stt = STTModule(on_text_callback=self.handle_stt) if isSTT else None
        print(f"[YOMI] STT : 준비완료 ({isSTT})")
        if isTTS:
            self.vits_model = VITS()
            self.tts_server = TTSServer(self.vits_model)
            print(f"[YOMI] TTSServer : 준비완료 ({isTTS})")
        self.tts = TTSClient(on_done=self.on_tts_done, on_start=self.on_tts_start) if isTTS else None
        print(f"[YOMI] TTSClient : 준비완료 ({isTTS})")
        self.llm = LLMClient() if isLLM else None
        print(f"[YOMI] LLM : 준비완료 ({isLLM})")
        self.VisionFace = VisonFaceMain(interval=2, on_vision_callback=self.handle_vision, viewGUI=True) if isVisionFace else None
        print(f"[YOMI] VisionFace : 준비완료 ({isVisionFace})")
        
        #ROS
        try:
            rosgraph.Master('/checker').getSystemState()
        except:
            print("[YOMI] roscore 연결 불가")
        #ROS Publisher
        if not rospy.core.is_initialized():
            rospy.init_node('yomi_core', anonymous=True)
        self.tts_state = rospy.Publisher('/tts_state', Bool, queue_size=10)
        print("[YOMI] ROS Publisher: 토픽(tts_state) 연결완료")

        #ROS Subscriber
        rospy.Subscriber('/switch_1_state', Bool, self.handle_switch, callback_args=1)
        rospy.Subscriber('/switch_2_state', Bool, self.handle_switch, callback_args=2)
        rospy.Subscriber('/switch_3_state', Bool, self.handle_switch, callback_args=3)
        rospy.Subscriber('/switch_4_state', Bool, self.handle_switch, callback_args=4)
        rospy.Subscriber('/switch_5_state', Bool, self.handle_switch, callback_args=5)
        rospy.Subscriber('/switch_6_state', Bool, self.handle_switch, callback_args=6)
        print("[YOMI] ROS Subscriber: 토픽(switch_state) 연결완료")
        

    def start(self):
        """모듈 시작"""
        if self.isSTT:
            threading.Thread(target=self._sttStart, daemon=True).start()
            self.sttTimer = threading.Timer(self.stt_timeout, self._vision_open)
            self.sttTimer.start()
        if self.isTTS:
            threading.Thread(target=self.tts_server.run, daemon=True).start()
        if self.isLLM:
            threading.Thread(target=self.llm.connect, daemon=True).start()
        if self.isVisionFace:
            self.VisionFace.run()

    def stop(self):
        """모듈 종료"""
        if self.isSTT: 
            self.stt.stop()
        if self.isTTS: 
            pass
        if self.isLLM:
            self.llm.disconnect()
        if self.isVisionFace: 
            self.VisionFace.stop()
        print("[YOMI] 모든 모듈 종료")

    def _sttStart(self):
        """STT모듈 멀티스레드로 시작 후, 1초마다 실행여부 판단"""
        self.stt.start()
        while True:
            time.sleep(1)
            self._sttResume()

    def _sttPause(self):
        """"STT 마이크 비활성화"""
        if self.isSTT:
            self.stt.micOff()

    def _sttResume(self):
        """STT 마이크 활성화"""
        if self.joy_master_flag and self.isSTT and not self.is_tts_running and self.sttEnable and not self.switch_flag and not self.vision_flag:
            self.stt.micOn()
    
    def handle_stt(self, stt_texts):
        """STT발생 시 처리 함수"""
        self.sttEnable = False
        self._sttPause()
        stt_text, stt_info = stt_texts
        print(f"[YOMI] (handle_stt) STT 결과: {stt_texts}")

        #STT Timeout
        if self.sttTimer:
            self.sttTimer.cancel()

        stt_prompt = f"청각정보 : {stt_text} \n"
        llm_prompt = stt_prompt + self.make_yolo_prompt()
        print(f"[YOMI] (handle_stt) llm_prompt :{llm_prompt}")
        self.handle_llm(llm_prompt)
        
        self.sttTimer = threading.Timer(self.stt_timeout, self._vision_open)
        self.sttTimer.start()
    
    def _vision_open(self):
        self.visionEnable = True
        #todo : 여기서 고개 돌아다니면서 확인하는 모션 함수 추가하면 될듯

    def handle_vision(self, visionText=None):
        """STT N초 이상 안들어오면 vision 정보 활용"""
        self.lastVision = visionText
        
        if self.joy_master_flag and self.isVisionFace and self.visionEnable:
            print("[YOMI] (handle_vision) STT timeout - vision 실행")
            print(f"[YOMI] (handle_vision) 감지 객체 :  {self.lastVision}")
            
            self.visionEnable=False
            self.vision_flag = True
            self._sttPause()

            self.handle_llm(self.make_yolo_prompt())

            #STT Timeout
            if self.sttTimer:
                self.sttTimer.cancel()
            self.sttTimer = threading.Timer(self.stt_timeout, self._vision_open)
            self.sttTimer.start()

    def handle_switch(self, msg, index): 
        """ROS토픽 이용 Switch 처리함수"""
        state = True if msg.data else False
        if state:
            touch_map = {
                0: "등",
                1: "왼팔",
                2: "오른팔",
                3: "왼손",
                4: "오른손",
                5: "머리"
            }
            if index in touch_map:
                print(f"[YOMI] (handle_switch) 눌린 부위: {touch_map[index]} (switch={index}) \n")
                
                self.switch_flag = True
                self._sttPause()
                
                switch_prompt = f"스위치 입력 : {touch_map[index]} 부위가 눌림 \n"
                llm_input = switch_prompt + self.make_yolo_prompt()
                self.handle_llm(llm_input)
                
            else:
                print(f"[YOMI] (handle_switch) 알 수 없는 switch 값: {index}")
    
    def handle_llm(self, text):
        responseAndEmotion = self.llm.send(text)
        if self.isTTS:
            self.try_send_tts(responseAndEmotion)


    def on_tts_start(self):
        """TTS가 시작될 때 호출되는 콜백 함수"""
        """팀장이 만들어라해서 TTS담당이 만들긴 했는데, 생각해보니 try_send_tts에서 할거 다하면 이게 굳이 필요가 없어 보인다,,, 쩝;"""
        pass
        # self.tts_state.publish(True)
        # self.is_tts_running = True

    def on_tts_done(self):
        """TTS가 끝날 때 호출되는 콜백 함수"""
        self.tts_state.publish(False)
        self.is_tts_running = False
        self.sttEnable = True
        self.switch_flag = False
        self.vision_flag = False

    def try_send_tts(self, response_text):
        print(f"[YOMI] (try_send_tts) TTS에서 받은 text: {response_text}")
        """TTS 상태 확인 후 텍스트 전송"""
        if self.isTTS and not self.is_tts_running:
            self.is_tts_running = True
            self.tts_state.publish(True)
            self.tts.send_text(response_text)
        else:
            print("[YOMI] (try_send_tts) TTS가 아직 끝나지 않았습니다. 새 요청 무시.")
    
    def make_yolo_prompt(self):
        """이미지 정보를 문자열로"""
        if not self.lastVision:
            return "시각정보 : 감지된 것이 없음"
        
        result = []
        for item in self.lastVision:
            label = item['label']
            box = item['box']
            center_x = (box[0] + box[2])/2
            center_y = (box[2] + box[3]) / 2
            result.append(f"시각정보 : {label}이 화면의 ({center_x:.1f}, {center_y:.1f})에 있습니다.")
        return ', '.join(result)

if __name__ == "__main__":
    service = Yomi(isSTT=True, isLLM=True, isTTS=True, isVisionFace=True)

    service.start()

    try:
        while not rospy.is_shutdown():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n 종료 중...")
    finally:
        service.stop()