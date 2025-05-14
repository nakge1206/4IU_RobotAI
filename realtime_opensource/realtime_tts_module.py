# tts_module.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import socket
import threading
import time
from queue import Queue
from RealtimeTTS.engines import EdgeEngine
from RealtimeTTS import TextToAudioStream


class TTSHandler:
    """TTS 요청을 큐로 받아 순차적으로 처리하는 비동기 재생 핸들러"""
    def __init__(self, voice="ko-KR-SoonBokNeural"):
        self.engine = EdgeEngine()
        self.engine.set_voice(voice)
        self.stream = TextToAudioStream(self.engine)

        self.queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

        self.warm_up()

    def warm_up(self):
        """초기 재생 지연 제거"""
        self.stream.feed("...")
        self.stream.play_async()
        if self.stream.play_thread:
            self.stream.play_thread.join(timeout=5)

    def _process_queue(self):
        while True:
            text, conn = self.queue.get()
            try:
                self.stream.feed(text)
                self.stream.play_async()
                if self.stream.play_thread:
                    self.stream.play_thread.join(timeout=10)
                time.sleep(0.3)
                conn.sendall(b"done")
            except Exception as e:
                print(" TTS 처리 중 오류:", e)
                try:
                    conn.sendall(b"fail")
                except:
                    pass
            finally:
                conn.close()

    def enqueue(self, text, conn):
        self.queue.put((text, conn))


class TTSServer:
    """TTS TCP 서버"""
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.tts = TTSHandler()

    def handle_client(self, conn, addr):
        try:
            data = conn.recv(1024)
            if not data:
                conn.close()
                return
            text = data.decode('utf-8').strip()
            print(f" 받은 문장: {text}")
            self.tts.enqueue(text, conn)
        except Exception as e:
            print(" 클라이언트 처리 오류:", e)
            conn.close()

    def start(self):
        print(" TTS 서버 실행 중 (대기 중...)")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()


class TTSClient:
    """외부 모듈에서 호출하는 클라이언트"""
    def __init__(self, host='127.0.0.1', port=65432, on_done=None):
        self.host = host
        self.port = port
        self.on_done = on_done

    def send_text(self, text: str):
        def _send():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(15)  # 안전하게 타임아웃
                    s.connect((self.host, self.port))
                    s.sendall(text.encode('utf-8'))

                    done_signal = s.recv(1024).decode()
                    if done_signal.strip() == "done" and self.on_done:
                        self.on_done()
            except Exception as e:
                print(" TTSClient 오류:", str(e))
                if self.on_done:
                    self.on_done()

        threading.Thread(target=_send, daemon=True).start()


#  테스트 실행
if __name__ == "__main__":
    server = TTSServer()
    server.start()
