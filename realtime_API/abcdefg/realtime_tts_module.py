import os, json, base64, websocket, threading
import simpleaudio as sa
from dotenv import load_dotenv
load_dotenv()

class TTSClient:
    def __init__(self, on_done=None):
        self.on_done = on_done
        self.ws = None

        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        self.headers = [
            "Authorization: Bearer " + os.getenv("OPENAI_API_KEY"),
            "OpenAI-Beta: realtime=v1"
        ]

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def send_text(self, text):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            event = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "instructions": text
                }
            }
            self.ws.send(json.dumps(event, ensure_ascii=False))

    def on_open(self, ws):
        print("✅ TTS WebSocket 연결 완료")

    def on_message(self, ws, message):
        data = json.loads(message)
        if data["type"] == "response.audio.delta":
            audio_data = base64.b64decode(data["delta"])
            sa.play_buffer(audio_data, 1, 2, 24000)
        elif data["type"] == "response.done":
            print("🔊 TTS 완료")
            if self.on_done:
                self.on_done()

    def on_error(self, ws, error):
        print("TTS 에러:", error)

    def on_close(self, ws, code, msg):
        print(f"TTS 연결 종료: {msg}")
