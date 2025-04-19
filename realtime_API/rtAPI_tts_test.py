import os, json, base64, websocket, threading
import simpleaudio as sa
from dotenv import load_dotenv
import time


# .env에서 API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WebSocket 연결 정보
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

# TTS 요청할 문장
INPUT_TEXT = "안녕하세요, 무엇을 도와드릴까요?"

class TTSClient:
    def __init__(self):
        self.ws = None
        self.text_queue = []

    def send_text(self, text):
        if self.ws and self.ws.sock.connected:
            event = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "instructions": text
                }
            }
            self.ws.send(json.dumps(event, ensure_ascii=False))
        else:
            print("TTS 서버와 연결되지 않음")
    
    def on_open(self, ws):
        print("TTS 연결 성공")
    
    def on_message(self, ws, message):
        try:
            event = json.loads(message)
            if event["type"] == "response.audio.delta":
                audio_data = base64.b64decode(event["delta"])
                wave_obj = sa.WaveObject(audio_data, 1, 2, 24000)
                wave_obj.play()
            elif event["type"] == "response.done":
                print("TTS 응답 완료")
        except Exception as e:
            print("TTS 처리 오류:", e)

    def on_close(self, ws, code, msg):
        print("TTS 연결 종료:", msg)

    def on_error(self, ws, error):
        print("TTS 에러:", error)

    def connect(self):
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_close=self.on_close,
            on_error=self.on_error
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        time.sleep(1)  # 연결 안정화 시간 확보