from RealtimeSTT import AudioToTextRecorder
import threading

class STTWrapper:
    def __init__(self, on_text_callback, model="base", language="ko", print_transcription_time=True, enable_realtime_transcription=True, use_main_model_for_realtime=True):
        self.on_text_callback = on_text_callback
        self.recorder = AudioToTextRecorder(
            model=model, 
            language=language, 
            enable_realtime_transcription=enable_realtime_transcription,
            use_main_model_for_realtime=use_main_model_for_realtime,
            print_transcription_time=print_transcription_time,
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