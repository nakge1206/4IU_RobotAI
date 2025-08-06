# yomi_core/llm_core/inference_client_ax4.py "ws://172.27.223.128:8765" localhost:8765
import asyncio
import websockets
import threading

class LLMClient:
    def __init__(self, uri="ws://172.27.238.28:8765"):
        self.uri = uri
        self.websocket = None
        
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()
    
    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def connect(self):
        future = asyncio.run_coroutine_threadsafe(self._connect(), self.loop)
        return future.result()

    async def _connect(self):
        self.websocket = await websockets.connect(
            self.uri,
            subprotocols=["llm-protocol"]
        )
        print(f"[LLMClient] 서버 연결됨: {self.uri}")

    def disconnect(self):
        if self.websocket:
            future = asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop)
            return future.result()

    async def _disconnect(self):
        if self.websocket:
            await self.websocket.close()
            print("[LLMClient] 연결 종료")
    
    def send(self, text):
        if self.websocket is None:
            self.connect()
        # print(text)
        future = asyncio.run_coroutine_threadsafe(self._send(text), self.loop)
        return future.result()
    
    async def _send(self, text):
        await self.websocket.send(text)
        response = await self.websocket.recv()
        return response

    async def chat_loop(self):
        self.connect()
        try:
            while True:
                msg = input("질문 (종료: ㅂㅂ): ")
                if msg.lower() in ['ㅂㅂ', 'exit', 'quit']:
                    break
                res = self.send(msg)
                print("로봇 응답:", res)
        except Exception as e:
            print("[클라이언트 오류]", e)
        finally:
            self.disconnect()


# 비동기 루프 진입
if __name__ == "__main__":
    client = LLMClient()
    asyncio.run(client.chat_loop())
