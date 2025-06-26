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
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))

# 각 모듈 임포트
from realtime_opensource.realtime_stt_module import STTModule
from realtime_opensource.realtime_tts_module import TTSClient, TTSServer  # TTS 연동 시 사용
# from robot_core.inference_koalpaca_12B import LLMResponder
from vision.ROD_module import YoloModule
from llm_core.gpt_fine_tuning_model import FineTunedGPTClient 



class Yomi:
    def __init__(self, isSTT=True, isTTS=True, isLLM=True, isVision=True):
        self.results = []
        self.is_tts_running = False
        self.lastVision = None

        #어쩔마스터플래그
        self.master_flag = True

        self.stt = STTModule(on_text_callback=self.handle_stt) if isSTT else None
        print("STT : 실행 준비 완료")

        self.tts_server = TTSServer()
        print("TTSServer : 준비완료")
        threading.Thread(target=self.tts_server.run_in_thread, daemon=True).start()
        self.tts = TTSClient(on_done=self.done_tts, on_start=self.start_tts) if isTTS else None  # TTS 사용 시
        print("TTSClient : 준비완료")

        # self.llm = LLMResponder() if isLLM else None
        self.llm = FineTunedGPTClient() if isLLM else None # gpt 사용시
        self.vision = YoloModule(interval=2, on_vision_callback=self.handle_vision, viewGUI=False) if isVision else None
        self.lastVision = None
        self.isVision = False

        #ros
        if not rospy.core.is_initialized():
            rospy.init_node('yomi_core', anonymous=True)
        self.llm_emotion = rospy.Publisher('/llm_emotion', String, queue_size=10)
        self.tts_state = rospy.Publisher('/tts_state', Bool, queue_size=10)

    def start(self):
        if self.stt:
            self.stt.start()
        if self.tts:
            pass
            # self.tts.connect() 
        if self.vision:
            self.isVision = True
            self.vision.start()
        print("yomi_core 시스템 준비완료...")

    def stop(self):
        if self.stt: 
            self.stt.stop()
        # if self.llm: 
        #   self.llm.stop() 추후 추가할 예정
        if self.tts: 
            self.tts.stop()
        if self.vision: 
            self.vision.stop()
        print("모든 모듈 종료")
    
    def pause(self):
        if self.stt:
            self.stt.pause()
        self.isVision=False

    def resume(self):
        #마스터 플래그가 true야 돌아가도록
        if self.master_flag:
            if self.stt and not self.is_tts_running:
                self.stt.resume()
            self.isVision=True
    
    def handle_stt(self, stt_texts):
        """STTModule에서 text가 생성될 때 마다 이 코드가 실행됨"""
        self.is_tts_running = True
        self.pause()      
        try:
            print(f"\n STT 결과: {stt_texts}")                
            labels = [item['label'] for item in self.lastVision] if self.lastVision else None
            self.llm_promt(stt_texts, labels, isSTT=True, isVision=True)
            print("STT->LLM")
        except Exception as e:
            print(f"LLM 처리 중 오류 발생 : {e}")

    def start_tts(self):
        """tts가 시작되면 tts_state topic에 True값 전송"""
        self.tts_state.publish(True)

    def done_tts(self):
        """TTS가 끝날때 마다 이 코드가 실행됨"""
        self.tts_state.publish(False)
        self.is_tts_running = False
        if not self.master_flag:
            threading.Timer(2.0, self.done_tts).start()
        else:
            self.resume()

    def handle_vision(self, visionText):
        """vision이 감지될때 마다 이 코드가 실행됨"""
        if self.master_flag:
            if self.isVision:
                self.lastVision = visionText
                labels = [item['label'] for item in self.lastVision] if self.lastVision else None
                print(labels)
                # if random.random() < 0.1:
                #     self.llm_promt(None, labels, False, True)

    def llm_promt(self, sttTexts, visionText, isSTT=True, isVision=True):
        if not self.llm:
            if self.tts: #근데 tts는 켜져있을 때
                if sttTexts:
                    self.tts.send_text(sttTexts[0])
            return


        if isSTT and isVision: #STT가 들어올때(영상정보까지 포함됨).
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
                print("LLM->TTS")
                self.tts.send_text(response)
                self.llm_emotion.publish(emotion)
                # if not self.is_tts_running:
                #     self.is_tts_running = True  # 재생 중 플래그
                #     print("LLM->TTS")
                #     self.tts.send_text(response)
                # else:
                #     print("TTS가 아직 끝나지 않았습니다. 새 요청 무시.")
            return
        
        if not isSTT and isVision:
            print("YOLO->LLM, 시각정보 : ", visionText)
            user_prompt = self.llm.build_instruction_vision(visionText)

            emotion, response = self.llm.chat(user_prompt)
            print(f" LLM 결과: {emotion}")
            print(f" LLM 결과: {response}")

            if self.tts:
                if not self.is_tts_running:
                    self.is_tts_running = True
                    print("LLM->TTS")
                    self.tts.send_text(response)

        else: #llm 껐을때
            if self.tts: #근데 tts는 켜져있을 때
                self.tts.send_text(stt_text)

    
    def test_llm(self):
        if self.llm:
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