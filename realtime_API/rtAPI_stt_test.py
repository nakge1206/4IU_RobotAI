import os
import json
import websocket
from dotenv import load_dotenv
import sounddevice as sd
import time
import threading
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

CHUNK_SIZE = 512

class STTClient:
    def __init__(self, on_text_callback):
        self.ws = None
        self.audio_thread = None
        self.on_text_callback = on_text_callback
        self.running = False

    def send_audio(self, ws):
        try:
            def callback(indata, frames, time_info, status):
                if ws.sock and ws.sock.connected and self.running:
                    ws.send(indata.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)

            info = sd.query_devices(3)
            sample_rate = int(info['default_samplerate'])

            with sd.InputStream(
                device=3,
                channels=1,
                samplerate=sample_rate,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback
            ):
                while self.running and ws.keep_running:
                    time.sleep(0.1)
        except Exception as e:
            print("🎤 STT InputStream 예외 발생:", e)

    def on_open(self, ws):
        print("서버 연결 성공.")
        self.running = True
        self.audio_thread = threading.Thread(target=self.send_audio, args=(ws,), daemon=True)
        self.audio_thread.start()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "text" in data:
                print("인식된 텍스트:", data["text"])
                self.on_text_callback(data["text"])
        except Exception as e:
            print("STT 오류: ", e)

    def on_close(self, ws, close_status_code, close_msg):
         print(f"STT 연결 종료: code={close_status_code}, msg={close_msg}")

    def on_error(self, ws, error):
        print("STT 에러: ", error)

    def start(self):
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_close=self.on_close,
            on_error=self.on_error
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def pause(self):
        print("🔇 STT 일시정지")
        self.running = False

    def resume(self):
        print("🔊 STT 재개")
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.running = True
            self.audio_thread = threading.Thread(target=self.send_audio, args=(self.ws,), daemon=True)
            self.audio_thread.start()