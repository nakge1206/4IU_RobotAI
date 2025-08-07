import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import socket
import threading
from queue import Queue
import simpleaudio as sa
import numpy as np
import sounddevice as sd
import torch
import commons
import util
from concurrent.futures import ThreadPoolExecutor
from model import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def play_audio(audio, rate):
    sd.play(audio, samplerate=rate)
    sd.wait()

class VITS:
    def __init__(self):
        checkpoint = "/home/micca/catkin_ws/src/4IU_RobotAI/yomi_core/tts/checkpoints/G_212000.pth"
        config = "/home/micca/catkin_ws/src/4IU_RobotAI/yomi_core/tts/example/configs/korean.json"

        self.device = torch.device("cuda")
        
        print("[TTS] [VITS] Using device : ", self.device)

        self.hps = util.get_hparams_from_file(config)
        self.spk_count = self.hps.data.n_speakers
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).to(self.device)  #모델을 GPU에 강제 업로드

        _ = self.net_g.eval()
        _ = util.load_checkpoint(checkpoint, self.net_g, None)

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def infer(self, text, spk_id=0):
        stn_tst = self.get_text(text)
        with torch.no_grad():
            x_tst = stn_tst.to(self.device).unsqueeze(0)  # GPU에 올림
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            sid = torch.LongTensor([spk_id]).to(self.device)

            print("[TTS] [VITS] 디바이스 상태:", x_tst.device, x_tst_lengths.device, sid.device)

            audio = self.net_g.infer(
                x_tst, x_tst_lengths, sid=sid,
                noise_scale=.667, noise_scale_w=0.8, length_scale=1
            )[0][0, 0].data.cpu().float().numpy()  # 결과만 CPU로 이동

        return audio



class TTSHandler:
    def __init__(self, vits_model):
        self.vits = vits_model
        self.queue = Queue()
        threading.Thread(target=self._process_queue, daemon=True).start()

    def _process_queue(self):
        while True:
            text, conn = self.queue.get()
            try:
                audio = self.vits.infer(text)
                audio_int16 = np.int16(audio * 32767)
                sd.play(audio_int16, samplerate=self.vits.hps.data.sampling_rate)
                sd.wait()
                conn.sendall(b"done")
            except Exception as e:
                print("[TTS] [TTSHandler] : ", e)
                try:
                    conn.sendall(b"fail")
                except:
                    pass
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

    def enqueue(self, text, conn):
        self.queue.put((text, conn))


class TTSServer:
    def __init__(self, vits_model, host='127.0.0.1', port=65432, max_workers=10):
        self.host = host
        self.port = port
        self.handler = TTSHandler(vits_model)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"[TTS] [Server] {self.host}:{self.port} 서버 실행 중...")
            while True:
                try:
                    conn, addr = s.accept()
                    self._handle_client(conn, addr)
                except socket.timeout:
                    continue  # 무시하고 다시 accept 대기
                except Exception as e:
                    print(f"[TTS] [Server] 예외 발생: {e}")
                    continue

    def _handle_client(self, conn, addr):
        try:
            data = conn.recv(1024)
            if not data:
                print(f"[TTS] [Server] 연결 종료됨 : {addr}")
                conn.close()
                return
            text = data.decode('utf-8').strip()
            print(f"[TTS] [Server] 연결요청({addr}): '{text}'")
            self.handler.enqueue(text, conn)
        except Exception as e:
            print(f"[TTS] [Server] 클라이언트 처리 오류 ({addr}):", e)
            try:
                conn.close()
            except Exception:
                pass


class TTSClient:
    def __init__(self, host='127.0.0.1', port=65432, on_done=None, on_start=None):
        self.host = host
        self.port = port
        self.on_done = on_done
        self.on_start = on_start
        self.isRunning = True

    def send_text(self, text: str):
        if not self.isRunning:
            print("[TTS] [Client] 서버가 비활성 상태입니다.")
            return

        def _send():
            try:
                if self.on_start:
                    self.on_start()

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host, self.port))
                    s.sendall(text.encode('utf-8'))
                    result = s.recv(1024).decode()
                    if result.strip() == "done":
                        print("[TTS] [Client] 재생 완료")
                    if self.on_done:
                        self.on_done()
            except Exception as e:
                import traceback
                print("[TTS] [Client] 오류:", e)
                traceback.print_exc()
                if self.on_done:
                    self.on_done()

        threading.Thread(target=_send, daemon=True).start()

    def stop(self):
        self.isRunning = False

def run_vits_server():
    vits_model = VITS()
    server = TTSServer(vits_model)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("[TTS] 서버 종료")


if __name__ == "__main__":
    run_vits_server()
