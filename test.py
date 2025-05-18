# test.py

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import torch, gc
import threading

# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtime_opensource'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'robot_core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))

# 각 모듈 임포트
from realtime_opensource.realtime_stt_module import STTModule
from realtime_opensource.realtime_tts_module import TTSClient  # TTS 연동 시 사용
from robot_core.llm_inference import LLMResponder
from vision.ROD_module import YoloModule
from robot_core.gpt_fine_tuning_model import FineTunedGPTClient 




class Yomi:
    def __init__(self, isSTT=True, isTTS=True, isLLM=True, isVision=True):
        self.results = []
        self.is_tts_running = False
        self.stt = STTModule(on_text_callback=self.handle_stt) if isSTT else None
        self.tts = TTSClient(on_done=self.resume_stt) if isTTS else None  # TTS 사용 시

        # self.llm = LLMResponder() if isLLM else None
        self.llm = FineTunedGPTClient() if isLLM else None
        self.yolo = YoloModule(interval=2, on_vision_callback=self.handle_vision, viewGUI=True) if isVision else None

    def start(self):
        print("\n 시스템 실행 중...")
        if self.stt:
            # self.stt.start()
            threading.Thread(target=self.stt.start, daemon=True).start()
        if self.tts:
            pass
            # self.tts.connect() 
        if self.yolo:
            self.yolo.run_detection()
    
    def stop(self):
        self.stt.stop()
        # self.llm.stop() # 이 기능 아직 없던데 모듈화 과정에서 추가해야함.
        self.tts.stop()
        self.yolo.stop()

    def handle_stt(self, text_tuple):
        """STTModule에서 text가 생성될 때 마다 이 코드가 실행됨."""
        self.is_tts_running = True
        if self.is_tts_running:
            self.stt.pause()

        if self.stt:
            print(f"\n STT 결과: {text_tuple}")

            try:
                stt_text, metadata = text_tuple
                emotion = metadata.get("emotion", "")
                event = metadata.get("event", "")

                print(f" LLM 추론 진입 → 텍스트: '{stt_text}', 감정: '{emotion}', 이벤트: '{event}'")


                if self.llm:
                    # gsq 모델
                # response = self.llm.generate_response(
                    #     stt_text,
                    #     emotion=emotion,
                    #     event=event,
                    #     mbti="INFP"
                    # )
        
                # gpt 모델
                    user_prompt = self.llm.build_instruction(stt_text, emotion, event)
                    response = self.llm.chat(user_prompt)

                #######################################
                print(f" LLM 결과: {response}")
                if self.tts:
                    print("llm(true), tts(true)")
                    self.tts.send_text(response)


                else:
                    if self.tts:
                        print("llm(false), tts(true)")
                        self.tts.send_text(stt_text)

            except Exception as e:
                print(" LLM 처리 중 오류 발생:", str(e))
                if self.tts:
                    self.tts.send_text("응~ 무슨 말인지 잘 모르겠어!")

    def resume_stt(self):
        self.is_tts_running = False
        if self.stt:
            self.stt.resume()

    def handle_vision(self, visionText):
        print(visionText)

    def llm_promt(self):
        pass

if __name__ == "__main__":
    service = Yomi(isSTT=True, isLLM=True, isTTS=True, isVision=True)

    service.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n 종료 중...")
        Yomi.stop()
