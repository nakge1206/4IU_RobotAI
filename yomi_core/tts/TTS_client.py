from TTS_server import TTSClient

client = TTSClient()

while True:
    text = input("📥 변환할 문장을 입력하세요 ('exit' 입력 시 종료): ")
    if text.lower() == 'exit':
        client.stop()
        break
    client.send_text(text)