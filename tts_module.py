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
    """TTS 클라이언트 클래스"""
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port

    def send(self, text: str):
        """문장을 서버에 전송"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(text.encode('utf-8'))
        except ConnectionRefusedError:
            print("❌ TTS 서버가 켜져 있는지 확인하세요! (TTSServer.start 실행 필요)")

# 🧪 테스트 실행
if __name__ == "__main__":
    server = TTSServer()
    server.start()
