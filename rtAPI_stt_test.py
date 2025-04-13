# example requires websocket-client library:
# pip install websocket-client

import os
import json
import websocket
from dotenv import load_dotenv
import sounddevice as sd
import time
import threading
import numpy as np


# .env에 있는 OPENAI_API_KEY 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WebSocket URL 및 헤더
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

# 오디오 설정
SAMPLE_RATE = 16000
CHUNK_SIZE = 512


class STTClient:
    def __init__(self, on_text_callback):
        self.ws = None
        self.on_text_callback = on_text_callback

    #마이크 입력 전송
    def send_audio(self, ws):
        def callback(indata, frames, time_info, status):
            if ws.sock and ws.sock.connected:
                ws.send(indata.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=callback
        ):
            while ws.keep_running:
                time.sleep(0.1)

    #WebSocket 열렸을 때 실행
    def on_open(self, ws):
        print("서버 연결 성공.")
        threading.Thread(target=self.send_audio, args=(ws,), daemon=True).start()

    #서버로부터 메시지 수신
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "text" in data:
                print("인식된 텍스트:", data["text"])
        except Exception as e:
            print("STT 오류: ", e)

    #WebSocket 종료 시
    def on_close(ws, close_status_code, close_msg):
        print("STT 연결 종료: ", close_msg)

    #에러 처리
    def on_error(ws, error):
        print("STT 에러: ", error)

    # WebSocket 실행
    def start(self):
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.run_forever()