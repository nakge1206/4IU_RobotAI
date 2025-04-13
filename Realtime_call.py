# Realtime_call.py
from Realtime_tts import play_sentences

# 📂 텍스트 파일 경로
filepath = "C:/Users/jgb08/OneDrive/바탕 화면/practice.txt"

# 📖 문장들 읽기
with open(filepath, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file if line.strip()]

# 🔈 음성 재생 (warm-up 자동 포함됨)
play_sentences(sentences)
