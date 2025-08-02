# inference_client.py
import asyncio
import websockets

async def interactive_mode():
    uri = "ws://172.27.244.83:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            text = input("질문 (종료하려면 'ㅂㅂ'): ")
            if text.lower() in ['ㅂㅂ', 'exit', 'quit']:
                print("클라이언트 종료.")
                break

            await websocket.send(text)           # 질문 보냄
            result = await websocket.recv()      # 응답 받음
            print(f"로봇 응답: {result}\n")

if __name__ == "__main__":
    asyncio.run(interactive_mode())
