import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os


class LLMResponder:
    def __init__(self,
                 model_path="yomi_core/llm_core/adapter_ax",
                 adapter_path=None):  # LoRA adapter 따로 분리된 경우만 사용
        print("[LLMResponder] 초기화 중...")

        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Model 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        # LoRA 어댑터 연결 (별도 존재할 경우)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("[LLMResponder] 모델 준비 완료")

    def wrap_prompt(self, text):
        """프롬프트 템플릿"""
        return f"### 사용자\n{text}\n\n### 로봇 (유아 말투로 대답해줘):"

    def generate_response(self, text):
        """텍스트를 입력받아 유아 말투로 응답 생성"""
        prompt = self.wrap_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        end = time.time()
        print(f"[LLMResponder] 추론 시간: {end - start:.2f}초")

        # 디코딩 후 프롬프트 제거
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):]

        return output_text.strip()

    def test(self):
        """직접 입력받아 LLM 응답 확인하는 CLI"""
        print("[LLMResponder] 테스트 모드 시작 (종료: ㅂㅂ)")
        while True:
            text = input("질문: ")
            if text.strip().lower() in ["ㅂㅂ", "exit", "quit"]:
                print("종료됨.")
                break
            response = self.generate_response(text)
            print("로봇 응답:", response)


# #단독 실행 시 CLI 테스트 진입
# if __name__ == "__main__":
#     llm = LLMResponder(
#         model_path="yomi_core/llm_core/adapter_ax",  # 모델+토크나이저 위치
#         adapter_path=None  # 이미 LoRA 통합 모델이면 None
#     )
#     llm.test()
