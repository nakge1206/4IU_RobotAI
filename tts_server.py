# tts_server.py
import socket
from RealtimeTTS.engines import EdgeEngine
from RealtimeTTS import TextToAudioStream
import time

# TTS 초기화
engine = EdgeEngine()
engine.set_voice("ko-KR-SoonBokNeural")
stream = TextToAudioStream(engine)

# 워밍업
stream.feed("..")
stream.play_async()
if stream.play_thread:
    stream.play_thread.join()

print("🟢 TTS 서버 실행 중 (문장을 수신하면 즉시 재생합니다)")

# 서버 소켓 설정
HOST = '127.0.0.1'
PORT = 65432
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024)
            if not data:
                continue
            text = data.decode('utf-8').strip()
            print(f"📥 받은 문장: {text}")
            stream.feed(text)
            stream.play_async()
            if stream.play_thread:
                stream.play_thread.join()
            time.sleep(0.3)  # 자연스러운 흐름