# service.py
from realtime_stt_test import STTWrapper
from tts import TTSClient
import threading

results = []

# 플래그로 STT 제어
stt_controller = None

def handle_stt(text):
    print(f"🎤 사용자 발화 인식: {text}")
    results.append(text)

    # STT 중단 후 TTS 실행
    stt_controller.stop()

    def speak_and_resume():
        tts.send_text(text)

    threading.Thread(target=speak_and_resume, daemon=True).start()

def resume_stt():
    stt_controller.start()

if __name__ == "__main__":
    print("시스템 실행 중...")

    # TTS 생성 (끝나면 STT 재개)
    tts = TTSClient(on_done=resume_stt)
    tts.connect()

    # STT 시작
    stt_controller = STTWrapper(on_text_callback=handle_stt)
    stt_controller.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("종료 중...")