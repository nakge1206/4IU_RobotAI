from RealtimeSTT import AudioToTextRecorder
import threading

class STTWrapper:
    def __init__(self, on_text_callback, model="base", language="ko"):
        self.recorder = AudioToTextRecorder(model=model, language=language)
        self.on_text_callback = on_text_callback
        self.running = False
        self.thread = None
    #Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
    # tiny : 0.8초 지연, 정확도는 썩음
    # base : 1.5초 지연, 정확도는 썩 좋지 않음
    # small : 4초 지연, 정확도는 꽤 좋음

    def _run(self):
        while self.running:
            self.recorder.text(self.on_text_callback)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

# results = []
# def process_text(text):
#     print("인식 결과:", text)
#     results.append(text)  # 리스트에 저장

# if __name__ == '__main__':
#     print("Wait until it says 'speak now'")
#     recorder = AudioToTextRecorder(model="base", language="ko")
    
    
#     try:
#         while True:
#             recorder.text(process_text)
#     except KeyboardInterrupt:
#         print("전체 저장 내용:")
#         print("\n".join(results))