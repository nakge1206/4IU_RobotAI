# inference_server.py

import asyncio
import websockets
import os
import sys

# 경로 추가: llm_responder.py와 emotion_classifier.py가 같은 폴더에 있다고 가정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_responder import LLMResponder
from emotion_classifier import classify_emotion

# 어댑터 경로 (로컬에 맞게 수정 가능)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "../../robot_core/gsq_lora_adapter_koalpaca")

# 모델 로딩
responder = LLMResponder(
    model_path="beomi/KoAlpaca-Polyglot-12.8B",
    adapter_path=ADAPTER_PATH
)

# WebSocket 요청 처리 함수
async def handle(websocket):
    async for message in websocket:
        print(f"입력 받음: {message}")
        try:
            # 감정 분류 먼저 수행
            emotion = classify_emotion(message)

            # 분류된 감정 기반 응답 생성
            response = responder.generate_response(
                stt_text=message,
                emotion=emotion,
                event="대화",
                mbti="INFP"
            )

            # 포맷 맞춰 응답
            final_output = f"감정: {emotion}\n대답: {response}"
        except Exception as e:
            print(f"오류 발생: {e}")
            final_output = "감정: 기쁨\n대답: 응~ 무슨 말인지 잘 모르겠어!"

        await websocket.send(final_output)
        print(f"응답 전송 완료: {final_output}")

# 서버 실행 루프
async def main():
    print("LLM Inference Server 실행 중 (포트 8765)...")
    async with websockets.serve(handle, "0.0.0.0", 8765):
        await asyncio.Future()  # 서버 유지

# 단독 실행 테스트용
if __name__ == "__main__":
    asyncio.run(main())
