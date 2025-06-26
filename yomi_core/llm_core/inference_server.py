# inference_server.py
import asyncio
import websockets
import os
import sys

# llm_responder 임포트용 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_responder import LLMResponder

# 현재 위치 기준 어댑터 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "../../robot_core/gsq_lora_adapter_koalpaca")

# 모델/어댑터 초기화
responder = LLMResponder(
    model_path="beomi/KoAlpaca-Polyglot-12.8B",
    adapter_path=ADAPTER_PATH
)

async def handle(websocket):
    async for message in websocket:
        print(f"입력 받음: {message}")
        try:
            response = responder.generate_response(
                stt_text=message,
                emotion="평소",
                event="대화",
                mbti="INFP"
            )
        except Exception as e:
            print(f"오류 발생: {e}")
            response = "이해 못했어."

        await websocket.send(response)
        print(f"응답 전송 완료: {response}")

async def main():
    print("LLM Inference Server 실행 중 (포트 8765)...")
    async with websockets.serve(handle, "0.0.0.0", 8765):
        await asyncio.Future()  # 서버 계속 유지

if __name__ == "__main__":
    asyncio.run(main())

