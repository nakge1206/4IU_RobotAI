# yomi_core/llm_core/inference_client_ax4.py
import asyncio
import websockets

class LLMClient:
    def __init__(self, uri="ws://127.0.0.1:8765"):
        self.uri = uri
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(
            self.uri,
            subprotocols=["llm-protocol"]
        )
        print(f"[LLMClient] 서버 연결됨: {self.uri}")

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            print("[LLMClient] 연결 종료")

    async def send(self, text):
        await self.websocket.send(text)
        response = await self.websocket.recv()
        return response

    async def chat_loop(self):
        await self.connect()
        try:
            while True:
                msg = input("질문 (종료: ㅂㅂ): ")
                if msg.lower() in ['ㅂㅂ', 'exit', 'quit']:
                    break
                res = await self.send(msg)
                print("로봇 응답:", res)
        except Exception as e:
            print("[클라이언트 오류]", e)
        finally:
            await self.disconnect()


# 비동기 루프 진입
if __name__ == "__main__":
    client = LLMClient()
    asyncio.run(client.chat_loop())
