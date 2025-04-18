from RealtimeSTT import AudioToTextRecorder
import threading

class STTWrapper:
    def __init__(self, on_text_callback):
        self.on_text_callback = on_text_callback
        self.recorder = AudioToTextRecorder(
            model="base", 
            language="ko", 
            print_transcription_time=True, #지연율 출력
            enable_realtime_transcription=False, #입력 중에도 결과 업데이트
            use_main_model_for_realtime=False, #실시간 적용을 보조모델이 아닌, 메인모델 사용
            on_realtime_transcription_stabilized=self.on_text_callback
        )
        self.running = False
        self.thread = None
    #Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
    # tiny : 0.8초 지연, 정확도는 썩음
    # base : 1.5초 지연, 정확도는 썩 좋지 않음
    # small : 4초 지연, 정확도는 꽤 좋음

    def _run(self):
        while self.running:
            self.recorder.text()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False