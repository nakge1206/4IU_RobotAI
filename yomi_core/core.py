import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import threading
import random
import time
#ros통신용
import rospy
from std_msgs.msg import String, Bool


# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtime_opensource'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision_face'))

# 각 모듈 임포트
from stt.realtime_stt_module import STTModule #STT
# from realtime_opensource.realtime_tts_module import TTSClient, TTSServer  # TTS 연동 시 사용
from tts.TTS_server import TTSClient, TTSServer  # TTS 연동 시 사용
# from llm_core.inference_koalpaca_12B import LLMResponder # LLM
# from robot_core.inference_koalpaca_12B import LLMResponder
from vision_face.ROD_module import YoloModule #Vision
from llm_core.gpt_fine_tuning_model import FineTunedGPTClient #GPT LLM



class Yomi:
    def __init__(self, isSTT=True, isTTS=True, isLLM=True, isVision=False):
        self.results = [] #
        self.is_tts_running = False
        self.lastVision = None

        self.isSTT = isSTT
        self.isTTS = isTTS
        self.isLLM = isLLM
        self.isVision = isVision


        #조이스틱 제어 플래그
        self.joy_master_flag = True

        self.stt = STTModule(on_text_callback=self.handle_stt) if isSTT else None
        print(f"STT : 준비완료 ({isSTT})")

        self.tts_server = TTSServer()
        print(f"TTSServer : 준비완료 ({isTTS})")
        
        threading.Thread(target=self.tts_server.run_in_thread, daemon=True).start()
        self.tts = TTSClient(on_done=self.done_tts, on_start=self.start_tts) if isTTS else None  # TTS 사용 시
        # self.tts = None
        print(f"TTSClient : 준비완료 ({isTTS})")

        # self.llm = LLMResponder() if isLLM else None
        self.llm = FineTunedGPTClient() if isLLM else None # gpt 사용시
        print(f"LLM : 준비완료 ({isLLM})")

        self.vision = YoloModule(interval=2, on_vision_callback=self.handle_vision, viewGUI=True) if isVision else None
        print(f"YOLO : 준비완료 ({isVision})")
        

        #ROS Publisher
        if not rospy.core.is_initialized():
            rospy.init_node('yomi_core', anonymous=True)
        self.llm_emotion = rospy.Publisher('/llm_emotion', String, queue_size=10)
        self.tts_state = rospy.Publisher('/tts_state', Bool, queue_size=10)
        print("ROS : 토픽(감정, tts실행여부) 연결완료")

    def start(self):
        if self.isSTT:
            threading.Thread(target=self.sttStart, daemon=True).start()
        if self.isTTS:
            pass
            # self.tts.connect() 
        if self.isVision:
            self.vision.start()

    def stop(self):
        if self.isSTT: 
            self.stt.stop()
        if self.isTTS: 
            pass
            # self.tts.stop()
        if self.isVision: 
            self.vision.stop()
        print("모든 모듈 종료")

    def sttStart(self):
        self.stt.start()
        while True:
            time.sleep(1)
            self.resume()

    
    def pause(self):
        """"STT 및 Vision 일시정지"""
        if self.isSTT:
            self.stt.micOff()
        self.isVision=False

    def resume(self):
        if self.joy_master_flag and not self.is_tts_running and self.isSTT:
            self.stt.micOn()
            
        self.isVision=True
    
    def handle_stt(self, stt_texts):
        """STTModule에서 text가 생성될 때 마다 이 코드가 실행됨"""
        self.is_tts_running = True
        self.pause()      
        try:
            print(f"\n STT 결과: {stt_texts}")                
            labels = [item['label'] for item in self.lastVision] if self.lastVision else None
            self.llm_promt(stt_texts, labels)
        except Exception as e:
            print(f"LLM 처리 중 오류 발생 : {e}")



    def handle_vision(self, visionText):
        """vision이 감지될때 마다 이 코드가 실행됨"""
        if self.joy_master_flag and self.isVision:
            self.lastVision = visionText
            labels = [item['label'] for item in self.lastVision] if self.lastVision else None
            print(labels)
            # if random.random() < 0.1:
            #     self.llm_promt(None, labels, False, True)

    def llm_promt(self, sttTexts, visionText):
        if not self.isLLM:
            self.is_tts_running = False
            self.resume()
            if self.tts: #근데 tts는 켜져있을 때
                if sttTexts:
                    pass
                    # self.tts.send_text(sttTexts[0])
            return

        if self.isSTT and self.isVision: #STT가 들어올때(영상정보까지 포함됨).
            # gsq 모델
            # response = self.llm.generate_response(
            #         stt_text,
            #         emotion=emotion,
            #         event=event,
            #         mbti="INFP"
            #     )

            # gpt 모델
            stt_text, metadata = sttTexts
            stt_emotion = metadata.get("stt_emotion", "")
            event = metadata.get("event", "")
            user_prompt = self.llm.build_instruction(stt_text, stt_emotion, event, visionText)

            emotion, response = self.llm.chat(user_prompt)
            print(f" LLM_emotion 결과: {emotion}")
            print(f" LLM 결과: {response}")
            self.llm_emotion.publish(emotion)

            if self.tts:
                if self.tts and not self.is_tts_running:
                    self.is_tts_running = True
                    print("LLM->TTS")
                    self.start_tts()
                    self.tts.send_text(response)
                else:
                    print("TTS가 아직 끝나지 않았습니다. 새 요청 무시.")
            return
        
        if not self.isSTT and self.isVision:
            print("YOLO->LLM, 시각정보 : ", visionText)
            user_prompt = self.llm.build_instruction_vision(visionText)

            emotion, response = self.llm.chat(user_prompt)
            print(f" LLM 결과: {emotion}")
            print(f" LLM 결과: {response}")

            if self.tts:
                if not self.is_tts_running:
                    self.is_tts_running = True
                    print("LLM->TTS")
                    # self.tts.send_text(response)

        else: #llm 껐을때
            if self.tts: #근데 tts는 켜져있을 때
                pass
                # self.tts.send_text(stt_text)

    
    def test_llm(self):
        if self.isLLM:
            # gsq 모델
            # response = self.llm.generate_response(
            #         stt_text,
            #         emotion=emotion,
            #         event=event,
            #         mbti="INFP"
            #     )
            
            # gpt 모델
            stt_text = "안녕. 밥 먹었어?"
            emotion = "행복"
            event = "말하는 중"
            visionText = None

            user_prompt = self.llm.build_instruction(stt_text, emotion, event)
            response = self.llm.generate_response(
                stt_text,
                emotion=emotion,
                event=event,
                mbti="INFP"
            )

            print(f" LLM 결과: {response}")
        else: #llm 껐을때
            pass



    def send_tts(self, text):
        """TTS 서버에 텍스트를 전송"""
        if self.tts:
            self.tts.send_text(text)        

    def start_tts_publish(self):
        """TTS가 시작되면 tts_state topic에 True값 전송"""
        self.tts_state.publish(True)

    def done_tts_publish(self):
        """TTS가 끝나면 tts_state topic에 False값 전송"""
        self.tts_state.publish(False)

    def try_send_tts(self, response_text):
        """TTS 상태 확인 후 텍스트 전송"""
        if self.tts and not self.is_tts_running:
            self.is_tts_running = True
            print("LLM->TTS")
            self.start_tts_publish()
            self.send_tts(response_text)
        else:
            print("TTS가 아직 끝나지 않았습니다. 새 요청 무시.")

    def on_tts_start(self):
        """TTS가 시작될 때 호출되는 콜백 함수"""
        self.start_tts_publish()
        self.is_tts_running = True

    def on_tts_done(self):
        """TTS가 끝날 때 호출되는 콜백 함수"""
        self.done_tts_publish()
        self.is_tts_running = False



    # def start_tts(self):
    #     """tts가 시작되면 tts_state topic에 True값 전송"""
    #     self.tts_state.publish(True)

    # def done_tts(self):
    #     """TTS가 끝날때 마다 이 코드가 실행됨"""
    #     self.tts_state.publish(False)
    #     self.is_tts_running = False
    #     if not self.joy_master_flag:
    #         threading.Timer(2.0, self.done_tts).start()
    #     else:
    #         self.resume()


if __name__ == "__main__":
    service = Yomi(isSTT=True, isLLM=True, isTTS=True, isVision=False)

    service.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n 종료 중...")
    finally:
        service.stop()