from realtime_stt_module import RealtimeSTTClient
from realtime_tts_module import TTSClient
import time

class ConversationService:
    def __init__(self):
        self.tts = TTSClient(on_done=self.resume_stt)
        self.stt = RealtimeSTTClient(on_text_callback=self.handle_stt)
        self.results = []
        self.is_tts_running = False

    def handle_stt(self, text):
        print(f"🎙️ STT 결과: {text}")
        self.results.append(text)
        #self.stt.stop()
        self.is_tts_running = True
        self.tts.send_text(text)

    def resume_stt(self):
        print("🎤 STT 재시작")
        self.is_tts_running = False
        self.stt.start()

    def start(self):
        print("🟢 시스템 시작됨.")
        self.tts.connect()
        self.stt.start()

if __name__ == "__main__":
    service = ConversationService()
    service.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("👋 종료 요청됨.")
        service.stt.stop()
        if service.tts.ws:
            service.tts.ws.close()
