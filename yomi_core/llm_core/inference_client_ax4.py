import asyncio
import websockets
import threading
import json
import time

class LLMClient:
    def __init__(self, uri="wss://pseudophilosophical-unextendedly-allan.ngrok-free.dev"):
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
        retries = 0
        while retries < 30:
            try:
                self.websocket = await websockets.connect(
                    self.uri,
                    subprotocols=["llm-protocol"]
                )
                print(f"[LLMClient] 서버 연결됨: {self.uri}")
                return
            except websockets.exceptions.ConnectionClosedError as e:
                retries += 1
                print(f"[LLMClient] 연결 실패. 3초 후 재시도")
                time.sleep(1)
        print(f"[LLMClient] 최대 재시도 횟수 도달. 서버 연결 실패.")

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
        payload = {"text": text, "mbti": mbti}
        future = asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)
        return future.result()

    def send_special(self, text):
        if self.websocket is None:
            self.connect()
        payload = text 
        future = asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)
        return future.result()
    
    async def _send(self, payload):
        message_to_send = ""
        if isinstance(payload, dict):
            # 1. 딕셔너리면 -> JSON으로 변환 (일반 채팅)
            message_to_send = json.dumps(payload, ensure_ascii=False)
        else:
            # 2. 딕셔너리가 아니면 -> String 그대로 사용 (특수 신호)
            message_to_send = str(payload) # 혹시 모르니 str() 처리
        try:
            await self.websocket.send(message_to_send)
            response = await self.websocket.recv()
            return response
        except websockets.exceptions.ConnectionClosedError:
            print("[LLMClient] 연결 끊김. 재연결 시도 중...")
            self.connect()

            await self.websocket.send(message_to_send)
            response = await self.websocket.recv()
            return response


    async def chat_loop(self):
        self.connect()
        try:
            while True:
                msg = input("질문 (종료: ㅂㅂ): ").strip()
                if msg.lower() in ['ㅂㅂ', 'exit', 'quit']:
                    break
                if msg.lower() in ['special']:
                    aa = input("String 문자 : ").strip()
                    res = self.send_special(aa)
                    print("로봇 응답:", res)
                    continue
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
