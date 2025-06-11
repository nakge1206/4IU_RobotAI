import os
import re
import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class LLMResponder:
    def __init__(self, model_path="beomi/KoAlpaca-Polyglot-12.8B", adapter_path="./gsq_lora_adapter_koalpaca"):
        # 오프로드 폴더 준비
        self.offload_dir = "C:\\Users\\COM\\Desktop\\yomi^_^\\4IU_RobotAI\\robot_core\\offload"
        os.makedirs(self.offload_dir, exist_ok=True)

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 기본 모델 로드 (offload_folder 추가)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # GPU에 최대한 올리고 나머지 CPU
            device_map={"": 0}        # 모든 레이어를 GPU 0번에 올리도록 강제
        )

        # LoRA 어댑터 연결
        peft_config = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
        self.model = PeftModel.from_pretrained(
            model=base_model,
            model_id=adapter_path,
            peft_config=peft_config,
            is_trainable=False
        )
        self.model.eval()

    def build_instruction(self, stt_text, emotion, event):
        return f"{stt_text.strip()}. 감정은 '{emotion}'이고 상황은 '{event}'야. 로봇은 유아 말투로 짧고 따뜻하게 반응해줘."

    def generate_response(self, stt_text, emotion, event, mbti="INFP"):
        start_time = time.time()

        instruction = self.build_instruction(stt_text, emotion, event)
        prompt = f"""{instruction}

### 사용자 ({mbti})
MBTI: {mbti}

### 로봇 (유아 역할, 한국어 반말로 대답해줘. 높임말은 쓰지 마. 영어 절대 금지):
"""
        # print(" Prompt:\n", prompt) #체크  # [변경] 디버깅용 주석 처리

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            inputs.pop("token_type_ids", None)  #추가, 모델이 안 쓰는 항목 제거
            inputs = inputs.to(self.model.device)  # 이후 GPU로 올림

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.6,
                    top_k=30,
                    top_p=0.8,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.decode(
                outputs[0].cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            # print("\n 디코딩 결과:\n", decoded)  # 디버깅용  # [변경] 주석 처리

            #  순수 로봇 응답만 추출
            if "### 로봇 (유아 역할, 한국어 반말로 대답해줘. 높임말은 쓰지 마):" in decoded:
                response = decoded.split("### 로봇 (유아 역할, 한국어 반말로 대답해줘. 높임말은 쓰지 마):")[-1].strip()
            else:
                response = decoded

            response = response.strip().splitlines()[-1].strip()

        except Exception as e:
            # print(" LLM 처리 오류:", str(e))  # [변경] 디버깅용 주석 처리
            response = "응~ 무슨 말인지 잘 모르겠어!"
            outputs = None

        response_clean = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,?!~]", "", response)

        # GPU 메모리 정리
        del inputs
        if outputs is not None:
            del outputs
        torch.cuda.empty_cache()
        gc.collect()

        # elapsed = time.time() - start_time
        # print(" 대답 완성. 소요시간:", round(elapsed, 2), "초")  # [변경] 출력 생략
        return response_clean

#  단독 실행 테스트용 main 함수
if __name__ == "__main__":
    llm = LLMResponder()
    while True:
        stt_text = input("할 말 적어.")
        emotion = "평소"
        event = "이야기"
        mbti = "INFP"
        if(stt_text == "ㅂㅂ"):
            break
        response = llm.generate_response(stt_text, emotion, event, mbti)
        print("LLM: ", response)  # [변경] 최종 출력만 표시

    
