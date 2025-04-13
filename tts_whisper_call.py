from tts_whisper import play_text_as_speech


# 파일 열기
with open("C:/Users/jgb08/OneDrive/바탕 화면/practice.txt", 'r', encoding='utf-8') as file:
    content = file.read()
    play_text_as_speech(content)
