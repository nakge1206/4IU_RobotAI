# text_to_speech.py
from openai import OpenAI
from IPython.display import Audio

client = OpenAI()

def play_text_as_speech(text: str, speed: float = 1.1, voice: str = "alloy", model: str = "tts-1") -> Audio:
    """
    주어진 텍스트를 음성으로 변환하여 Jupyter에서 재생합니다.

    Parameters:
    - text (str): 변환할 텍스트
    - speed (float): 말하기 속도 (기본값 1.1)
    - voice (str): 사용할 음성 (기본값 "alloy")
    - model (str): TTS 모델 이름 (기본값 "tts-1")

    Returns:
    - IPython.display.Audio 객체
    """
    speech_file_path = "tts_audio.mp3"
    response = client.audio.speech.create(
        model=model,
        input=text,
        voice=voice,
        response_format="mp3",
        speed=speed,
    )
    response.stream_to_file(speech_file_path)
    return Audio(speech_file_path)
