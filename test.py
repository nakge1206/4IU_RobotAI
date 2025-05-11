# test.py

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import torch, gc

# 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtime_opensource'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'robot_core'))

# STT/LLM 모듈 임포트
from realtime_opensource.realtime_stt_module import STTWrapper
from realtime_opensource.realtime_tts_module import TTSClient  # TTS 연동 시 사용
from robot_core.llm_inference import LLMResponder


class ConversationService:
    def __init__(self):
        self.results = []
        self.is_tts_running = False
        self.tts = TTSClient(on_done=self.resume_stt)  # TTS 사용 시
        self.stt = STTWrapper(on_text_callback=self.handle_stt)
        self.llm = LLMResponder()

    def start(self):
        print("\n시스템 실행 중...")
        # self.tts.connect()
        self.stt.start()

    def handle_stt(self, text_tuple):
        self.is_tts_running = True
        if self.is_tts_running:
            self.stt.pause()

        print(f"\nstt 결과: {text_tuple}")

        stt_text, metadata = text_tuple
        emotion = metadata.get("emotion", "")
        event = metadata.get("event", "")
        
        response = self.llm.generate_response(
            stt_text,
            emotion=emotion,
            event=event,
            mbti="INFP"
        )

        gc.collect()
        torch.cuda.empty_cache()

        print(f"\nllm 결과: {response}")
        self.tts.send_text(response)  # TTS 연동 시

    def resume_stt(self):
        self.is_tts_running = False
        self.stt.resume()


if __name__ == "__main__":
    service = ConversationService()
    service.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n종료 중...")
