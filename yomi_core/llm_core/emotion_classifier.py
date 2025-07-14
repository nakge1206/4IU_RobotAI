# emotion_classifier.py

from transformers import pipeline

# 모델 로딩
classifier = pipeline("text-classification", model="dlckdfuf141/korean-emotion-kluebert-v2")

# 원래 모델의 라벨 → 목표 8가지 감정으로 매핑
label_map = {
    '공포': '공포',
    '놀람': '놀라움',
    '분노': '분노',
    '슬픔': '슬픔',
    '중립': '기대',      # 또는 '신뢰'로 바꿔도 됨
    '행복': '기쁨',
    '혐오': '혐오',
    # 혹시 예외 라벨 나오면 기본값
}
DEFAULT_EMOTION = '기쁨'

def classify_emotion(text: str) -> str:
    try:
        result = classifier(text, top_k=1)[0]  # 가장 높은 확률의 감정 하나 선택
        raw_label = result['label']
        return label_map.get(raw_label, DEFAULT_EMOTION)
    except Exception as e:
        print(f"[감정 분류 오류] {e}")
        return DEFAULT_EMOTION

# 단독 실행 테스트
if __name__ == "__main__":
    while True:
        user_input = input("입력 문장: ")
        if user_input.lower() in ['exit', 'quit', 'ㅂㅂ']:
            break
        emotion = classify_emotion(user_input)
        print(f"감정: {emotion}")
