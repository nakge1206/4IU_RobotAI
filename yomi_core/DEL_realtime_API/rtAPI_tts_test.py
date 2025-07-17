import os, json, base64, websocket, threading
import simpleaudio as sa
from dotenv import load_dotenv
import time
import io
import wave

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

class TTSClient:
    def __init__(self, on_done=None):
        self.ws = None
        self.on_done = on_done
        self.audio_chunks = []

    def set_on_done_callback(self, callback):
        self.on_done = callback

    def send_text(self, text):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.audio_chunks.clear()
            event = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio"],
                    "instructions": text
                }
            }
            self.ws.send(json.dumps(event, ensure_ascii=False))
        else:
            print("❌ TTS 서버와 연결되지 않음")

    def on_open(self, ws):
        print("✅ TTS WebSocket 연결 완료")

    def on_message(self, ws, message):
        try:
            event = json.loads(message)
            if event["type"] == "response.audio.delta":
                audio_data = base64.b64decode(event["delta"])
                self.audio_chunks.append(audio_data)

            elif event["type"] == "response.audio.done":
                print("🔊 오디오 수신 완료, 재생 시작")
                self.play_audio()

            elif event["type"] == "response.done":
                print("✅ TTS 응답 완료")
                if self.on_done:
                    self.on_done()

        except Exception as e:
            print("❌ TTS 처리 오류:", e)

    def play_audio(self):
        try:
            audio_bytes = b''.join(self.audio_chunks)
            with io.BytesIO(audio_bytes) as audio_stream:
                wave_obj = sa.WaveObject.from_wave_read(wave.open(audio_stream, 'rb'))
                play_obj = wave_obj.play()
                play_obj.wait_done()
        except Exception as e:
            print("❌ 오디오 재생 오류:", e)

    def on_close(self, ws, code, msg):
        print(f"🔌 TTS 연결 종료 (code={code}, msg={msg})")

    def on_error(self, ws, error):
        print("❌ TTS WebSocket 오류:", error)

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
        time.sleep(1)  # 안정적인 연결을 위한 대기