import time
import DEL_yomi_motor_core
import yomi_motor.scripts.DEL_yomi_motion as DEL_yomi_motion
from DEL_yomi_motor_core import motorCore
from yomi_motor.scripts.DEL_yomi_motion import MotionController

class motor_delay:
    def __init__(self):
        self.motor_core = motorCore()
        self.motor_controller = MotionController()

    def handle_motor_llm(self, text):
        """모터 관련 LLM 처리 (전체 지연율 측정)"""
        start_time = time.time()
        print(f"[YOMI] (handle_motor_llm) MOTOR_LLM 실행됨")

        # 1. LLM 기반 모터 명령 얻기
        func_name = self.motor_core.ask_finetuned_model(text)
        print(f"[YOMI] (handle_motor_llm) MOTOR_LLM 결과 : {func_name}")
        current_time = time.time()
        Lcurrent_time = current_time - start_time
        print(f"LLM 지연 시간: {Lcurrent_time: 2f}초")

        # 2. 모터 동작 실행
        if hasattr(self.motor_controller, func_name):
            func = getattr(self.motor_controller, func_name)
            print(f"[YOMI] [handle_motor_llm] '{func_name}' 함수 실행.")
            if callable(func):
                func()
            else:
                print(f"[YOMI] [handle_motor_llm] Warning: '{func_name}'은 함수가 아닙니다.")
        else:
            print(f"[YOMI] [handle_motor_llm] Warning: MotionController에 '{func_name}' 함수 없음.")
        end_time = time.time()
        LMcurrent_time = end_time - current_time
        print(f"동작 끝 시간: {LMcurrent_time: 2f}초")

        # 3. 마무리 처리
        self.motor_controller.finger_end()

        # 전체 처리 시간 계산
        latency = end_time - start_time
        print(f"[YOMI] (handle_motor_llm) 전체 처리 지연율: {latency:.2f}초")
        return latency


def build_prompt():
    """사용자로부터 순차적으로 입력받아 프롬프트 문자열 생성"""
    emotion = input("emotion: ")
    visionDetect = input("visionDetect: ")
    switchState = input("switchState (예: 0 0 0 0 0 0): ")
    mbti = input("MBTI: ")
    speech = input("speech: ")
    response = input("response: ")

    # 포맷 맞추기
    prompt = (
        f"emotion: {emotion}\n"
        f"visionDetect: {visionDetect}\n"
        f"switchState: {switchState}\n"
        f"MBTI: {mbti}\n"
        f"speech: {speech}\n"
        f"response: {response}"
    )
    return prompt


if __name__ == "__main__":
    motor = motor_delay()

    try:
        while True:
            print("\n[프롬프트 입력 모드] (exit 입력 시 종료)\n")
            prompt = build_prompt()

            if prompt.lower().startswith("emotion: exit"):
                print("[STOP] 프로그램을 종료합니다.")
                break

            latency = motor.handle_motor_llm(prompt)
            print(f"[LOOP] 실행 지연율: {latency:.4f}초\n")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STOP] 사용자가 강제 종료했습니다.")
