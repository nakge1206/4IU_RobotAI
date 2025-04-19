# service.py
from realtime_stt_module import STTWrapper
from realtime_tts_module import TTSClient
import threading

'''
git알려주기위해 일부러 수정한 부분.
나중에 누가 본다면 지워서 git 연습해보길~
'''

'''
이건 준서한테 알려주는 git 주석임
나중에 한번 시험삼아 해보길~
'''

class ConversationService:
    def __init__(self):
        # 결과 저장용
        self.results = []
        self.is_tts_running = False
        self.tts_lock = threading.Lock()
        # TTS가 끝나면 STT 다시 시작하는 콜백 연결
        self.tts = TTSClient(on_done=self.resume_stt)
        # STT에서 텍스트를 받으면 handle_stt 실행
        self.stt = STTWrapper(on_text_callback=self.handle_stt)

    def start(self):
        print("\n시스템 실행 중...")
        self.tts.connect()
        self.stt.start()

    def handle_stt(self, text):
        with self.tts_lock:
            if self.is_tts_running:
                print(f"\n[무시됨] STT 결과 (TTS 중): {text}")
                return
            print(f"\n음성 인식: {text}")
            self.results.append(text)
            self.is_tts_running = True

        self.tts.send_text(text)

    def resume_stt(self):
        with self.tts_lock:
            print("TTS 완료 → STT 재시작")
            self.is_tts_running = False



if __name__ == "__main__":
    service = ConversationService()
    service.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n종료 중...")