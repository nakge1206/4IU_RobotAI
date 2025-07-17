from audio_recorder_realtime_v2 import RealtimeSTTClient
from rtAPI_tts_test import TTSClient
import time

class ConversationService:
    def __init__(self):
        self.results = []
        self.is_tts_running = False

        # TTS 설정: 끝났을 때 STT 재시작
        self.tts = TTSClient(on_done=self.resume_stt)

        # STT 설정: 텍스트 수신시 처리 콜백
        self.stt = RealtimeSTTClient(on_text_callback=self.handle_stt)

    def start(self):
        print("🟢 시스템 시작됨.")
        self.tts.connect()
        self.stt.start()

    def handle_stt(self, text):
        print(f"🎙️ STT 결과: {text}")
        self.is_tts_running = True
        self.stt.stop()  # 실시간이라면 pause도 가능
        self.results.append(text)
        self.tts.send_text(text)

    def resume_stt(self):
        print("🔁 TTS 완료, STT 재개.")
        self.is_tts_running = False
        self.stt.start()  # 다시 음성 인식 시작

if __name__ == "__main__":
    service = ConversationService()
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("👋 종료 요청됨.")
        service.stt.stop()