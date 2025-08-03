# yomi_core/llm_core/inference_server_ax4.py
import asyncio
import websockets
import os
import sys

sys.path.append(os.path.dirname(__file__))
from llm_responder_ax4 import LLMResponder

llm = LLMResponder(
    model_path="yomi_core/llm_core/adapter_ax",
    adapter_path=None
)

async def handle_connection(websocket):  # path 생략 가능 (websockets 12.x+)
    print("[LLM 서버] 클라이언트 연결됨")
    while True:
        try:
            question = await websocket.recv()
            print(f"[수신] 질문: {question}")

            response = llm.generate_response(question)
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
        subprotocols=["llm-protocol"],  # ✅ 중요: subprotocol 명시
        ping_interval=None,
        ping_timeout=None
    )
    print("[LLM 서버] 시작됨 (ws://0.0.0.0:8765)")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
