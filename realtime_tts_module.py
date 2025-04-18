# tts_module.py
import socket
import threading
import time
from RealtimeTTS.engines import EdgeEngine
from RealtimeTTS import TextToAudioStream

class TTSHandler:
    """TTS 스트리밍 엔진 초기화 및 재생 처리"""
    def __init__(self, voice="ko-KR-SoonBokNeural"):
        self.engine = EdgeEngine()
        self.engine.set_voice(voice)
        self.stream = TextToAudioStream(self.engine)
        self.warm_up()

    def warm_up(self):
        """초기 지연 제거용 워밍업"""
        self.stream.feed("..")
        self.stream.play_async()
        if self.stream.play_thread:
            self.stream.play_thread.join()

    def play(self, text: str):
        """문장을 TTS로 재생"""
        self.stream.feed(text)
        self.stream.play_async()
        if self.stream.play_thread:
            self.stream.play_thread.join()
        time.sleep(0.3)

class TTSServer:
    """TTS 서버 클래스"""
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.tts = TTSHandler()

    def handle_client(self, conn, addr):
        with conn:
            data = conn.recv(1024)
            if not data:
                return
            text = data.decode('utf-8').strip()
            print(f"📥 받은 문장: {text}")
            self.tts.play(text)

    def start(self):
        """TTS 서버 시작"""
        print("🟢 TTS 서버 실행 중 (문장을 수신하면 즉시 재생합니다)")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr)).start()

class TTSClient:
    def __init__(self, host='127.0.0.1', port=65432, on_done=None):
        self.host = host
        self.port = port
        self.on_done = on_done

    def connect(self):
        pass  # 현재는 사용 안 함 (연결은 send 시마다 새로 함)

    def send_text(self, text: str):
        def _send():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host, self.port))
                    s.sendall(text.encode('utf-8'))
                    # TTS 음성 길이만큼 대기 후 on_done 콜백 호출
                    time.sleep(len(text) * 0.1 + 0.5)  # 대략적으로 시간 예측
                    if self.on_done:
                        self.on_done()
            except ConnectionRefusedError:
                print("❌ TTS 서버가 켜져 있는지 확인하세요!")

        threading.Thread(target=_send, daemon=True).start()


        
# 🧪 테스트 실행
if __name__ == "__main__":
    server = TTSServer()
    server.start()

# 📤 외부에서 간단히 호출할 수 있도록 함수 제공
_client = TTSClient()

def play(text: str):
    """문장을 TTS 서버에 전송"""
    _client.send(text)