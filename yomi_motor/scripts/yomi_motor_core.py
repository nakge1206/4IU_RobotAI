import openai
from yomi_motion import MotionController
from dotenv import load_dotenv
import os
import time

# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_finetuned_model(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:your-org:your-model-name",  # 여기는 실제 fine-tune 이름으로
        messages=[
            {"role": "system", "content": "당신은 감정과 행동을 분류하는 AI입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def run_motion_from_gpt():
    controller = MotionController()

    while True:
        print("\n--- GPT 입력 포맷 예시 ---")
        print("emotion: disgust\nvisionDetect: none\nswitchState: 0 0 0 0 0 0\nMBTI: INFP\nspeech: 싫어! 가까이 가지 마.\nresponse: 멀리 가자.")
        print("exit 입력 시 종료됩니다.\n")

        user_input = input(">> GPT 입력:\n")
        if user_input.lower() == "exit":
            break

        try:
            func_name = ask_finetuned_model(user_input)
            print(f"[GPT 결과] 함수명: {func_name}")

            if hasattr(controller, func_name):
                func = getattr(controller, func_name)
                if callable(func):
                    func()
                else:
                    print(f"[ERROR] '{func_name}'은 함수가 아닙니다.")
            else:
                print(f"[ERROR] MotionController에 '{func_name}' 함수 없음.")
        except Exception as e:
            print(f"[예외 발생] {e}")

        time.sleep(1)

if __name__ == '__main__':
    run_motion_from_gpt()
