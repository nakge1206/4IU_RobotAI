'''
stt가 반환하는 result 값은
(text, {"language", "emotion", "event", "itn"})
'''
from audio_recorder_ko import AudioToTextRecorder

class STTModule:
    def __init__(self, on_text_callback):
        self.on_text_callback = on_text_callback
        self.recorder = AudioToTextRecorder(
            language="ko", 
            spinner = False, #마이크 입력 등에 대해 돌아가는 애니메이션 및 글자 출력 여부
            silero_sensitivity = 0.5, #silero의 민감도. 0이 둔감, 1이 민감
            silero_use_onnx = True, # PyTorch대신 onnx로 사용. 아무튼 빠르다고 함.
            silero_deactivity_detection=True, # silero를 발화 종료 감지에 사용
            print_transcription_time=True, #지연율 출력
            debug_mode = False, # 콘솔에 디버그 정보를 출력
            allowed_latency_limit = 100, #큐에 저장하는 최대 청크 수
            no_log_file = False # False면 디버그파일을 만들지 않음
        )

    def _run(self):
        result = self.recorder.text()
        if result:
            self.on_text_callback(result)

    def start(self):
        """ stt 실행 """
        self._run()

    def stop(self):
        """ stt 정지 """
        self.recorder.abort()

    def pause(self):
        """ stt 일시정지(마이크 비활성화) """
        self.recorder.set_microphone(False)

    def resume(self):
        """ stt 제개(마이크 활성화) """
        self.recorder.set_microphone(True)
        self._run()