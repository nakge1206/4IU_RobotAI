# tts_client.py
import socket

def play_with_tts(text: str):
    """TTS 서버에 문장을 전송해 음성 재생을 요청하는 함수"""
    HOST = '127.0.0.1'
    PORT = 65432
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(text.encode('utf-8'))
    except ConnectionRefusedError:
        print("❌ TTS 서버가 켜져 있는지 확인하세요! (tts_server.py 실행 필요)")