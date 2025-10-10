from TTS_server import TTSClient
import time

start = time.time()
stop = time.time()

def tts_done():
    print("tts_done 실행")
    global stop, start
    stop = time.time()
    curr = stop - start
    print(curr)

client = TTSClient(on_middle=tts_done)


while True:
    text = input("📥 변환할 문장을 입력하세요 ('exit' 입력 시 종료): ")
    if text.lower() == 'exit':
        client.stop()
        break

    start = time.time()
    client.send_text(text)