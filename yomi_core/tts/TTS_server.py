import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import socket
import threading
from queue import Queue
import simpleaudio as sa
import numpy as np

import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


class VITS:
    def __init__(self):
        # 경로를 고정으로 내장
        checkpoint = r"C:\\Users\\COM\\Desktop\\vits\\checkpoints\\logs\\korean_vits\\G_212000.pth"
        config = r"C:\\Users\\COM\\Desktop\\vits\\example\\configs\\korean.json"

        self.hps = utils.get_hparams_from_file(config)
        self.spk_count = self.hps.data.n_speakers
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).cuda()
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(checkpoint, self.net_g, None)

        # 서버 자동 실행
        self.server = TTSServer(self)


    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def infer(self, text, spk_id=0):
        stn_tst = self.get_text(text)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([spk_id]).cuda()
            audio = self.net_g.infer(
                x_tst, x_tst_lengths, sid=sid,
                noise_scale=.667, noise_scale_w=0.8, length_scale=1
            )[0][0, 0].data.cpu().float().numpy()
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
                play_obj = sa.play_buffer(audio_int16, 1, 2, self.vits.hps.data.sampling_rate)
                play_obj.wait_done()
                conn.sendall(b"done")
            except Exception as e:
                print("[TTS 처리 오류]:", e)
                try:
                    conn.sendall(b"fail")
                except:
                    pass
            finally:
                conn.close()

    def enqueue(self, text, conn):
        self.queue.put((text, conn))


class TTSServer:
    def __init__(self, vits_model, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.handler = TTSHandler(vits_model)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"[✓] VITS 서버 실행 중: {self.host}:{self.port}")
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    def _handle_client(self, conn, addr):
        try:
            data = conn.recv(1024)
            if not data:
                conn.close()
                return
            text = data.decode('utf-8').strip()
            print(f"[요청 수신] {addr}: '{text}'")
            self.handler.enqueue(text, conn)
        except Exception as e:
            print("[!] 클라이언트 처리 오류:", e)
            conn.close()


class TTSClient:
    def __init__(self, host='127.0.0.1', port=65432, on_done=None, on_start=None):
        self.host = host
        self.port = port
        self.on_done = on_done
        self.on_start = on_start
        self.isRunning = True

    def send_text(self, text: str):
        if not self.isRunning:
            print("TTSClient : 서버 비활성 상태입니다.")
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
                        print("✅ 재생 완료")
                    if self.on_done:
                        self.on_done()
            except Exception as e:
                print("TTSClient 오류:", e)
                if self.on_done:
                    self.on_done()

        threading.Thread(target=_send, daemon=True).start()

    def stop(self):
        self.isRunning = False


# ========== 서버 자동 실행 ==========
vits = VITS() 
