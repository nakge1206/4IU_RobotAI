# service.py
from realtime_stt_module import STTWrapper
from realtime_tts_module import TTSClient
import threading

'''
git알려주기위해 일부러 수정한 부분.
나중에 누가 본다면 지워서 git 연습해보길~
'''

class ConversationService:
    def __init__(self):
        # 결과 저장용
        self.results = []

        # TTS가 끝나면 STT 다시 시작하는 콜백 연결
        self.tts = TTSClient(on_done=self.resume_stt)

        # STT에서 텍스트를 받으면 handle_stt 실행
        self.stt = STTWrapper(on_text_callback=self.handle_stt)

    def start(self):
        print("🔄 시스템 실행 중...")
        self.tts.connect()
        self.stt.start()

    def handle_stt(self, text):
        print(f"🎤 사용자 발화 인식: {text}")
        self.results.append(text)

        # STT 중지
        self.stt.stop()

        # TTS로 말하게 하고 완료 시 resume_stt가 호출됨
        threading.Thread(target=lambda: self.tts.send_text(text), daemon=True).start()

    def resume_stt(self):
        print("🗣️ TTS 완료 → STT 재시작")
        self.stt.start()


if __name__ == "__main__":
    service = ConversationService()
    service.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("🛑 종료 중...")