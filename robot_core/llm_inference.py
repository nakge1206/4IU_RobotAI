import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import time


class LLMResponder:
    def __init__(self, model_path="Bllossom/llama-3.2-Korean-Bllossom-3B", adapter_path=None):
        # 1. 어댑터 경로 수동 설정 (지정된 절대 경로 사용)
        if adapter_path is None:
            adapter_path = "robot_core/gsq_lora_adapter"  # ✅ 윈도우 절대 경로 문자열

        # 2. 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = None

        # 3. 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 4. LoRA 어댑터 로드 (문자열 경로)
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

### 로봇 (유아 역할)
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,  # 생성할 최대 토큰 수
                do_sample=True, # 샘플링 여부(True: 확률기반, False: greedy)
                temperature=0.6,    # 샘플링 다양성(클수록 창의적, defualt = 1)
                top_k=30,   # 고려할 토큰 후보 수
                top_p=0.85,     # 누적 확률이 p 이하인 후보만 선택
                repetition_penalty=1.2,     # 반복 단어 억제 계수
                no_repeat_ngram_size=2,     # 같은 2-gram(단어 2개 조합) 반복 금지
                eos_token_id=self.tokenizer.eos_token_id    # 종료 토큰
            )

        decoded = self.tokenizer.decode(
            outputs[0].cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        response = decoded.split("### 로봇 (유아 역할)\n")[-1].strip().split("###")[0].strip()
        response_clean = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,?!~]", "", response)
        
        elapsed = time.time() - start_time
        print("대답 완성. 소요시간 : ", elapsed)
        return response_clean
