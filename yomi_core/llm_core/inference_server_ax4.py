
import asyncio
import websockets
import os
import sys
import json

sys.path.append(os.path.dirname(__file__))
from llm_responder_ax4 import LLMResponder

llm = LLMResponder(
    model_path="C:/Users/COM/Desktop/yomi/4IU_RobotAI/yomi_core/llm_core/adapter_ax",
    adapter_path=None
)

async def handle_connection(websocket):
    print("[LLM 서버] 클라이언트 연결됨")
    while True:
        try:
            raw_question = await websocket.recv()
            question, mbti_code = json.loads(raw_question)
            if mbti_code == "I":
                mbti = "INFP"
            elif mbti_code == "E":
                mbti = "ESTJ"
            else:
                mbti = None
            print(f"[수신] 질문: {question}")
            if mbti is not None:
                print(f"[수신] MBTI: {mbti}")
            else:
                print("[수신] MBTI 정보 없음")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, llm.generate_response, question, None, None, mbti)

            await websocket.send(response)
            print(f"[전송] 응답: {response}")
        except Exception as e:
            print(f"[서버 오류] {e}")
            await websocket.send("❌ 서버 내부 오류 발생")
            break

async def main():
    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
        subprotocols=["llm-protocol"],
        ping_interval=60,      # 수정
        ping_timeout=60        # 수정
    )
    print("[LLM 서버] 시작됨 (ws://0.0.0.0:8765)")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
