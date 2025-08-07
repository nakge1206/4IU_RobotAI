import openai
# from yomi_motion import MotionController
from dotenv import load_dotenv
import os
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'stt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'llm_core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision_face'))

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

yomi_STT_path = os.path.join(base_dir, 'stt')
yomi_TTS_path = os.path.join(base_dir, 'tts')
yomi_LLM_path = os.path.join(base_dir, 'llm_core')
yomi_vision_path = os.path.join(base_dir, 'vision_face')

from stt.realtime_stt_module import STTModule        # STT   
from tts.TTS_server import TTSServer, VITS, TTSClient # TTS
from vision_face.VisionFace import VisonFaceMain # VISION
from llm_core.inference_client_ax4 import LLMClient # LLM

class motorCore:
    def __init__(self):
        # 환경변수에서 API 키 가져오기
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def ask_finetuned_model(self, prompt: str) -> str:
        """Motor LLM 함수"""
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:personal::C1YKvB1S",  # 여기는 실제 fine-tune 이름으로
            messages=[
                {"role": "system", "content": "당신은 감정과 행동을 분류하는 AI입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

    def run_motion_from_gpt(self):
        """main에서 디버깅용으로 무한반복 하는 함수"""
        # controller = MotionController()

        while True:
            print("\n--- GPT 입력 포맷 예시 ---")
            print("emotion: disgust\nvisionDetect: none\nswitchState: 0 0 0 0 0 0\nMBTI: INFP\nspeech: 싫어! 가까이 가지 마.\nresponse: 멀리 가자.")
            print("exit 입력 시 종료됩니다.\n")

            user_input = input(">> GPT 입력:\n")
            if user_input.lower() == "exit":
                break

            # speech 항목 확인
            speech_value = None
            for line in user_input.split("\n"):
                if line.strip().lower().startswith("speech:"):
                    speech_value = line.split(":", 1)[1].strip()
                    break

            try:
                if not speech_value:  # speech가 없거나 비어있으면
                    func_name = "wait_command"
                    print("[INFO] speech가 감지되지 않아 'wait_command' 함수로 대체합니다.")
                else:
                    func_name = self.ask_finetuned_model(user_input)
                    print(f"[GPT 결과] 함수명: {func_name}")

            except Exception as e:
                print(f"[예외 발생] {e}")

            time.sleep(1)

if __name__ == '__main__':
    motor_core = motorCore()
    motor_core.run_motion_from_gpt()
