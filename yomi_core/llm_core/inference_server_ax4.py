# yomi_core/llm_core/inference_server_ax4.py
import asyncio, json, websockets
from llm_responder_ax4 import LLMResponder

responder = None

async def handle_connection(websocket):
    global responder
    print("[LLM 서버] 클라이언트 연결됨")
    try:
        async for message in websocket:
            print("[DEBUG] 수신 원문:", message)
            try:
                data = json.loads(message)
                if isinstance(data, list):  # 리스트 형태일 경우
                    data = {"text": data[0], "mbti": data[1] if len(data) > 1 else None}

                user_input = data.get("text", "")
                mbti = data.get("mbti", None)

                print(f"[수신] 질문: {user_input}")
                if mbti:
                    print(f"[수신] MBTI: {mbti}")

                response = responder.generate_response(user_input, mbti=mbti)

                await websocket.send(response)
                print(f"[전송] 응답: {response}")

            except Exception as e:
                print("[서버 오류]", e)
                if not websocket.closed:
                    await websocket.send("❌ 서버 내부 오류 발생")
    except websockets.exceptions.ConnectionClosedOK:
        print("[서버] 클라이언트 정상 종료")

async def main():
    global responder
    responder = LLMResponder()
    print("[LLM 서버] 시작됨 (ws://127.0.0.1:8765)")

    async with websockets.serve(handle_connection, "127.0.0.1", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
