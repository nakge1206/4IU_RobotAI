import os
import re
import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LLMResponder:
    def __init__(self, model_path="Bllossom/llama-3.2-Korean-Bllossom-3B", adapter_path=None):
        if adapter_path is None:
            adapter_path = "robot_core/gsq_lora_adapter"

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = None

        # 모델 및 어댑터 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
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

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.6,
                    top_k=30,
                    top_p=0.85,
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
            response = decoded.split("### 로봇 (유아 역할)\n")[-1].strip().split("###")[0].strip()

        except Exception as e:
            print(" LLM 처리 오류:", str(e))
            response = "응~ 무슨 말인지 잘 모르겠어!"
            outputs = None

        response_clean = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,?!~]", "", response)

        # GPU 메모리 정리
        del inputs
        if outputs is not None:
            del outputs
        torch.cuda.empty_cache()
        gc.collect()

        elapsed = time.time() - start_time
        print(" 대답 완성. 소요시간:", round(elapsed, 2), "초")
        return response_clean
