# tts_call.py
from tts_client import play_with_tts
import time

# 📂 텍스트 파일 경로
filepath = "C:/Users/jgb08/OneDrive/바탕 화면/practice.txt"

# 📖 파일 열어서 한 줄씩 문장 읽기
with open(filepath, "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip()]

# 🔊 한 문장씩 TTS 서버로 보내기
for sentence in lines:
    print(f"📤 전송 중: {sentence}")
    play_with_tts(sentence)
    time.sleep(0.4)  # 문장 사이 템포 조절