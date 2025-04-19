# service.py
from rtAPI_stt_test import STTClient
from rtAPI_tts_test import TTSClient
import time

def main():
    tts = TTSClient()
    tts.connect()

    def handle_stt_text(text):
        print(f"STT 인식 결과: {text}")
        tts.send_text(text)

    stt = STTClient(on_text_callback=handle_stt_text)
    stt.start()

if __name__ == "__main__":
    main()