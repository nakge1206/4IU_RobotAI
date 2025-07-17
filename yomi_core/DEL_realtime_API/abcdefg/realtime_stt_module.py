import os, json, base64, websocket, threading
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

class RealtimeSTTClient:
    def __init__(self, on_text_callback):
        self.on_text_callback = on_text_callback
        self.ws = None
        self.stream = None
        self.session_created = False
        self.running = False

        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        self.headers = [
            "Authorization: Bearer " + os.getenv("OPENAI_API_KEY"),
            "OpenAI-Beta: realtime=v1"
        ]

    def float_to_pcm16(self, float32_array):
        array = np.clip(float32_array, -1, 1)
        int16_array = (array * 32767).astype(np.int16)
        return int16_array.tobytes()

    def encode_audio(self, pcm16_bytes):
        return base64.b64encode(pcm16_bytes).decode("utf-8")

    def on_open(self, ws):
        print("🌐 WebSocket 연결됨")

        session_update_event = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                    "prompt": "",
                    "language": "ko"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "input_audio_noise_reduction": {
                    "type": "near_field"
                }
            }
        }

        ws.send(json.dumps(session_update_event))
        self.session_created = True
        self.running = True
        self.start_stream()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            print("📩 수신 이벤트:", data["type"])

            # if data["type"] == "conversation.item.input_audio_transcription.delta":
            #     partial = data.get("delta", "")
            #     if partial:
            #         print("📝 부분 인식:", partial)

            if data["type"] == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "")
                if transcript:
                    print("✅ 전체 인식:", transcript)
                    self.on_text_callback(transcript)

            elif data["type"] == "error":
                print("🚫 에러 발생:", data)

        except Exception as e:
            print("❌ STT 처리 오류:", e)

    def on_error(self, ws, error):
        print("❌ STT 에러:", error)

    def on_close(self, ws, code, msg):
        print(f"🔌 WebSocket 종료됨: code={code}, msg={msg}")
        self.stop_stream()

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def start_stream(self):
        def callback(indata, frames, time_info, status):
            if self.running and self.ws.sock and self.ws.sock.connected:
                pcm_data = self.float_to_pcm16(indata[:, 0])
                encoded = self.encode_audio(pcm_data)
                self.ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": encoded
                }))

        self.stream = sd.InputStream(callback=callback, samplerate=24000, channels=1)
        self.stream.start()
        print("🎧 오디오 캡처 중...")

    def stop_stream(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("🛑 STT 종료 요청")

    def stop(self):
        self.stop_stream()
        if self.ws:
            self.ws.close()
