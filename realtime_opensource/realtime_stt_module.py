'''

'''


from audio_recorder import AudioToTextRecorder
import time

class STTWrapper:
    def __init__(self, on_text_callback):
        self.on_text_callback = on_text_callback
        self.recorder = AudioToTextRecorder(
            language="ko", 
            print_transcription_time=True, #지연율 출력
            silero_deactivity_detection=True
        )

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