import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import socket
import threading
from queue import Queue
from gtts import gTTS
from pydub import AudioSegment
import io
import simpleaudio as sa


def speak_gtts(text, lang="ko"):
    """gTTS로 음성 생성 후 메모리 재생"""
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        play_obj = sa.play_buffer(audio.raw_data,
                                  num_channels=audio.channels,
                                  bytes_per_sample=audio.sample_width,
                                  sample_rate=audio.frame_rate)
        play_obj.wait_done()
    except Exception as e:
        print(" [gTTS 재생 오류]:", e)


class TTSHandler:
    """TTS 요청을 큐로 받아 순차적으로 처리하는 비동기 재생 핸들러"""
    def __init__(self, lang="ko"):
        self.lang = lang
        self.queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _process_queue(self):
        while True:
            text, conn = self.queue.get()
            try:
                speak_gtts(text, lang=self.lang)
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
            print(f"받은 문장: {text}")
            self.tts.enqueue(text, conn)
        except Exception as e:
            print("TTSServer : 클라이언트 처리 오류:", e)
            conn.close()

    def run_in_thread(self):
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(None)
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()


class TTSClient:
    """외부 모듈에서 호출하는 클라이언트"""
    def __init__(self, host='127.0.0.1', port=65432, on_done=None, on_start=None):
        self.host = host
        self.port = port
        self.on_done = on_done
        self.on_start = on_start
        self.isRunning = True

    def send_text(self, text: str):
        if not self.isRunning:
            print("TTSClient : tts 꺼져있음.")
            return

        def _send():
            try:
                if self.on_start:
                    self.on_start()
                
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(None)  
                    s.connect((self.host, self.port))
                    s.sendall(text.encode('utf-8'))

                    done_signal = s.recv(1024).decode()
                    if done_signal.strip() == "done" and self.on_done:
                        self.on_done()
            except Exception as e:
                print(" TTSClient 오류:", str(e))
                if self.on_done:
                    self.on_done()

        # _send()
        threading.Thread(target=_send, daemon=True).start()
    
    def stop(self):
        print("TTSClient 종료.")

