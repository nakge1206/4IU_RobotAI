from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import re

class motorCore:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        
    def ask_finetuned_model(self, prompt):
        print("[yomi_motor_core] (ask_finetuned_model) 실행됨.")
        try:
            character_info = (
                "당신은 감정과 행동을 분류하는 AI입니다."
                "사용자가 입력한 정보를 바탕으로 아래 함수 중 하나를 정확히 반환하십시오"
                #"또한 입력한 정보에서 MBTI가 I로 시작할시 I로 시작하는 함수에서 E로 시작힐시는 E로 시작하는 함수에서 찾으십시오."
                "함수: I_joy1, I_joy2, I_joy3,"
                "I_trust1, I_trust2, I_trust3,"
                "I_fear1, I_fear2, I_fear3,"
                "I_surprise1, I_surprise2, I_surprise3, "
                "I_sadness1, I_sadness2, I_sadness3,"
                "I_disgust1, I_disgust2, I_disgust3,"
                "I_anger1, I_anger2, I_anticipation1, I_anticipation2, I_anticipation3,"
                "E_joy1, E_joy2, E_joy3, E_trust1, E_trust2, E_trust3, E_fear1, E_fear2, E_fear3"
                "E_surprise1, E_surprise2, E_surprise3, E_sadness1, E_sadness2, E_sadness3"
                "E_disgust1, E_disgust2, E_disgust3, E_anger1, E_anger2"
                "E_anticipation1, E_anticipation2, E_anticipation3"
            )
            response = self.client.chat.completions.create(
                model="ft:gpt-3.5-turbo-1106:personal::C1YKvB1S",
                messages=[
                    {"role": "system", "content": character_info},
                    {"role": "user", "content": prompt}
                ],
                timeout=1.3
            )
            print("[yomi_motor_core] (ask_finetuned_model) 대답 받았음.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("[yomi_motor_core] (ask_finetuned_model) 오류 발생:", e)
            return "I_surprise1"

    def run_motion_from_gpt(self):
        while True:
            print("\n--- GPT 입력 포맷 예시 ---")
            print("emotion: disgust\nvisionDetect: none\nswitchState: 0 0 0 0 0 0\nMBTI: INFP\nspeech: 싫어! 가까이 가지 마.\nresponse: 멀리 가자.")
            print("exit 입력 시 종료됩니다.\n")

            user_input = input(">> GPT 입력:\n")
            if user_input.lower() == "exit":
                break

            # speech 항목 추출
            speech_match = re.search(r"speech:\s*(.*?)\s*response:", user_input, re.IGNORECASE)
            speech_value = speech_match.group(1).strip() if speech_match else None

            try:
                if not speech_value:
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
