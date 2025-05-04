'''
현재 문제점.
1. threading을 안쓰고 하자니, 마이크 입력을 받을때 다른 자원들 올 스탑임.
아니였음. thread 다뺴고 했는데도, 일시정지가 안됨.
-> 그냥 마이크자체를 멈춰버림.

2. threading을 쓰자니, 일시정지를 못하고 쓰레드 반환 후 재할당 해야해서 속도가 너무 느려지고
-> realtime opensource로 만드는거 좋은데, 이렇게 관리가 안되면 안된다고 판단. 베이스라인 변경해야할거 같음
= sense voice

04.21. sense voice를 small로 쓰자니 비회귀라 문장이 길어지면 성능이 떨어지고, large로 쓰자니 너무 느릴거 같음.
일단 small로 진행해보겠음.
'''


#from audio_recorder import AudioToTextRecorder
from audio_recorder import AudioToTextRecorder
import time

class STTWrapper:
    def __init__(self, on_text_callback):
        self.on_text_callback = on_text_callback
        self.recorder = AudioToTextRecorder(
            model="base", 
            language="ko", 
            print_transcription_time=True, #지연율 출력
            silero_deactivity_detection=True
        )
    #Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
    # tiny : 0.8초 지연, 정확도는 썩음
    # base : 1.5초 지연, 정확도는 썩 좋지 않음
    # small : 4초 지연, 정확도는 꽤 좋음

    def _run(self):
        # result, info = self.recorder.text()
        # '''
        # info의 내용
        # language
        # emotion
        # event
        # itn
        # '''
        # emotion = info.get("emotion") if info else "감정 비어있음"
        # event = info.get("event") if info else "이벤트 비어있음"
        # print(f"텍스트: {result}")
        # print(f"감정: {emotion}")
        # print(f"이벤트: {event}")
        result = self.recorder.text()
        print(result)
        if result:
            self.on_text_callback(result)

    def start(self):
        self._run()

    def stop(self):
        self.recorder.abort()

    def pause(self):
        self.recorder.set_microphone(False)

    def resume(self):
        self.recorder.set_microphone(True)
        self._run()