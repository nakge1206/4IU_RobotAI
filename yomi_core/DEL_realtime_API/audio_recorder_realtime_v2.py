import os, json, base64, websocket, threading
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
import time

class RealtimeSTTClient:
    def __init__(self, sample_rate=16000, chunk_size=512):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        self.headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1"
        ]
        self.ws = None
        self.stream = None
        self.connected = False

    def _float32_to_pcm16(self, float32_array):
        """Float32 -> PCM16 -> Base64"""
        int16_array = np.int16(np.clip(float32_array, -1, 1) * 32767)
        return base64.b64encode(int16_array.tobytes()).decode("utf-8")

    def _on_open(self, ws):
        self.connected = True
        print("🌐 WebSocket 연결됨")
        self._send_session_update()
        self._start_audio_stream()

    def _on_message(self, ws, message):
        event = json.loads(message)
        if event["type"] == "input_audio_buffer.speech_started":
            print("🗣️ 말 시작 감지")
        elif event["type"] == "input_audio_buffer.speech_stopped":
            print("🤐 말 끝 감지")
        elif event["type"] == "response.text.done":
            text = event["response"]["output"][0]["text"]
            print("✅ 인식된 텍스트:", text)

    def _on_close(self, ws, code, msg):
        self.connected = False
        print(f"🔌 WebSocket 종료됨: code={code}, msg={msg}")

    def _on_error(self, ws, error):
        print("❌ WebSocket 오류:", error)

    def _send_session_update(self):
        event = {
            "type": "session.update",
            "session": {
                "input_audio_format": {
                    "type": "linear_pcm",
                    "sample_rate": self.sample_rate
                },
                "turn_detection": {
                    "mode": "vad"
                }
            }
        }
        self.ws.send(json.dumps(event))

    def _audio_callback(self, indata, frames, time_info, status):
        if self.connected and self.ws and self.ws.sock and self.ws.sock.connected:
            b64_audio = self._float32_to_pcm16(indata[:, 0])
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_audio
            }))

    def _start_audio_stream(self):
        print("🎙️ InputStream 열기 시도...")
        self.stream = sd.InputStream(callback=self._audio_callback,
                                     samplerate=self.sample_rate,
                                     blocksize=self.chunk_size,
                                     channels=1,
                                     dtype='float32')
        self.stream.start()
        print("✅ InputStream 열림")

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.ws:
            self.ws.close()
        print("🛑 STT 종료 요청")

# 사용 예시
if __name__ == "__main__":
    stt = RealtimeSTTClient()
    stt.connect()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stt.stop()