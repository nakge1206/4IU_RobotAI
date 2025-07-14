# inference_client.py
import asyncio
import websockets

async def send_text(text):
    uri = "ws://172.27.244.83:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(text)
        result = await websocket.recv()
        print(f"서버 응답: {result}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python inference_client.py \"추론할 텍스트\"")
    else:
        asyncio.run(send_text(sys.argv[1]))