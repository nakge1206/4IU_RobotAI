# Realtime_tts.py
from RealtimeTTS.engines import EdgeEngine
from RealtimeTTS import TextToAudioStream
import time

# 🌟 전역 TTS 엔진 및 스트림 객체 생성 (프로그램 시작 시 딱 한 번만)
engine = EdgeEngine()
engine.set_voice("ko-KR-SoonBokNeural")
stream = TextToAudioStream(engine)

_warmed_up = False  # 내부 플래그: 워밍업 완료 여부

def warm_up_tts():
    """최초 한 번만 TTS 워밍업 수행"""
    global _warmed_up
    if not _warmed_up:
        stream.feed("..")
        stream.play_async()
        if stream.play_thread:
            stream.play_thread.join()
        _warmed_up = True

def play_sentences(sentences: list[str]):
    """여러 문장을 지연 없이 순차적으로 출력"""
    warm_up_tts()  # 사용 전 자동 워밍업 보장
    for sentence in sentences:
        stream.feed(sentence)
        stream.play_async()
        if stream.play_thread:
            stream.play_thread.join()
        time.sleep(0.3)  # 자연스러운 템포 조절
