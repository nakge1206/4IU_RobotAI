from RealtimeSTT import AudioToTextRecorder

results = []
def process_text(text):
    print("인식 결과:", text)
    results.append(text)  # 리스트에 저장

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(model="base", language="ko")
    #Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
    # tiny : 0.8초 지연, 정확도는 썩음
    # base : 1.5초 지연, 정확도는 썩 좋지 않음
    # small : 4초 지연, 정확도는 꽤 좋음
    
    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        print("전체 저장 내용:")
        print("\n".join(results))