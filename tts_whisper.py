# text_to_speech.py
from openai import OpenAI
import pygame
import threading

client = OpenAI()

def play_text_as_speech(text: str, speed: float = 1.1, voice: str = "alloy", model: str = "tts-1"):
    """
    주어진 텍스트를 음성으로 변환하여 음성을 바로 재생합니다.
    백그라운드 스레드에서 음성을 생성하고 재생합니다.

    Parameters:
    - text (str): 변환할 텍스트
    - speed (float): 말하기 속도 (기본값 1.1)
    - voice (str): 사용할 음성 (기본값 "alloy")
    - model (str): TTS 모델 이름 (기본값 "tts-1")
    """

    def generate_and_play_speech():
        # 음성 파일 생성
        speech_file_path = "tts_audio.mp3"
        response = client.audio.speech.create(
            model=model,
            input=text,
            voice=voice,
            response_format="mp3",
            speed=speed,
        )
        response.stream_to_file(speech_file_path)

        # pygame 초기화
        pygame.mixer.init()
        
        # MP3 파일 재생
        pygame.mixer.music.load(speech_file_path)
        pygame.mixer.music.play()

        # 음성이 끝날 때까지 기다리기
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    # 음성 생성 및 재생을 별도의 스레드에서 실행
    threading.Thread(target=generate_and_play_speech).start()
