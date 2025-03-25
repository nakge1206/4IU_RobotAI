#! python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

from gpt_query import ask_gpt
from tts_whisper import play_text_as_speech

global is_speaking
is_speaking = False

def on_tts_end():
    global is_speaking
    is_speaking = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="사용할 모델",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="마이크가 소리를 감지할 수 있는 에너지 임계값", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="녹음의 실시간성(주기)을 몇 초로 할지 설정", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "얼마 동안 소리가 없으면 새로운 문장으로 간주할지 설정", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="기본 마이크 이름 설정 (SpeechRecognition용). "
                                 "'list'를 넣으면 사용 가능한 마이크를 출력함.", type=str)
    args = parser.parse_args()

    # 마지막으로 오디오가 큐에서 꺼내진 시간
    phrase_time = None
    # 스레드 안전한 오디오 데이터 큐 (백그라운드 녹음 스레드와 메인 스레드 간 공유)
    data_queue = Queue()
    # SpeechRecognition을 통해 오디오 녹음을 설정 (자동으로 음성이 끝날 때를 감지해줌)
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # 이 설정을 꼭 해야 함. 동적 에너지 보정이 너무 민감해져서 감지가 멈추지 않을 수 있음
    recorder.dynamic_energy_threshold = False

    # 리눅스 사용자용 마이크 설정
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("사용 가능한 마이크 장치: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"\"{name}\" 이름의 마이크가 발견됨")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # 모델 로드 또는 다운로드
    model = args.model
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = [''] # 변환된 텍스트 저장 리스트

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        백그라운드에서 오디오 녹음이 끝나면 호출되는 콜백 함수
        audio: 녹음된 오디오 데이터 객체
        """
        # 원시 오디오 데이터를 가져와서 큐에 저장
        data = audio.get_raw_data()
        data_queue.put(data)

    # 백그라운드에서 오디오 데이터를 계속 수신하는 스레드 생성
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    

    # 모델 로드 완료 알림
    #print("Model loaded.\n")
    #print(f"모델 로드 완료. [{args.mode.upper()} 모드]로 시작합니다. 마이크에 말하세요.\n")
    #print(f"모델 로드 완료. [GPT 연동 모드]로 시작합니다. 마이크에 말하세요.\n")
    print(f"모델 로드 완료. [GPT 연동 + 음성 출력 모드]로 시작합니다. 마이크에 말하세요.\n")


    while True:
        try:
            now = datetime.utcnow()
            # 큐에 데이터가 있으면 꺼내서 처리
            if not data_queue.empty():
                phrase_complete = False
                # 이전 문장과 현재 녹음 간 간격이 설정된 timeout을 넘었으면 새로운 문장으로 간주
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                # 큐에 쌓인 오디오 데이터를 합치고 큐 초기화
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # 오디오 데이터를 모델이 이해할 수 있는 형태로 변환
                # 16비트 정수를 32비트 float로 변환하고, 32768로 정규화
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Whisper 모델을 사용해 음성을 텍스트로 변환
                result = audio_model.transcribe(audio_np, language='ko', fp16=torch.cuda.is_available())
                text = result['text'].strip()

                global is_speaking
                if text and not is_speaking:
                    print(f"\n[사용자] {text}")
                    gpt_response = ask_gpt(text)
                    print(f"[GPT] {gpt_response}\n")

                    # 음성으로 읽어주기
                    is_speaking = True
                    play_text_as_speech(gpt_response, on_complete=on_tts_end)
                
                # 문장 구분: 말 사이가 비었으면 새로운 문장 추가, 아니면 기존 문장 덮어쓰기
                # if phrase_complete:
                #     transcription.append(text)
                # else:
                #     transcription[-1] = text

                # # 콘솔 화면 지우고 텍스트 출력
                # os.system('cls' if os.name=='nt' else 'clear')
                # for line in transcription:
                #     print(line)
                # # Flush stdout.
                # print('', end='', flush=True)
            else:
                # CPU 과부하 방지용 대기
                sleep(0.25)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()