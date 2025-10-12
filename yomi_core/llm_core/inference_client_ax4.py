import asyncio
import websockets
import threading
import json

class LLMClient:
    def __init__(self, uri="ws://172.27.235.179:8765"):
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
    
    def send(self, text, mbti="INFP"):
        if self.websocket is None:
            self.connect()
        # ✅ 딕셔너리 형태로 전송
        payload = {"text": text, "mbti": mbti}
        future = asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)
        return future.result()
    
    async def _send(self, payload):
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))
        response = await self.websocket.recv()
        return response

    async def chat_loop(self):
        self.connect()
        try:
            while True:
                msg = input("질문 (종료: ㅂㅂ): ").strip()
                if msg.lower() in ['ㅂㅂ', 'exit', 'quit']:
                    break
                # 필요하면 매번 또는 최초 1회 MBTI 코드 입력
                mbti_in = input("MBTI 코드 입력 (I/E, 건너뛰기 Enter): ").strip().upper()
                payload = {"question": msg, "mbti_code": mbti_in}
                res = self.send(json.dumps(payload, ensure_ascii=False))
                print("로봇 응답:", res)
        except Exception as e:
            print("[클라이언트 오류]", e)
        finally:
            self.disconnect()


if __name__ == "__main__":
    client = LLMClient()
    asyncio.run(client.chat_loop())
