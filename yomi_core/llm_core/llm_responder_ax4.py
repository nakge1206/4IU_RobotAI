import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import re
from pathlib import Path


class LLMResponder:
    def __init__(self,
                 model_path="C:/Users/COM/Desktop/yomi/4IU_RobotAI/yomi_core/llm_core/adapter_ax",
                 adapter_path=None):
        print("[LLMResponder] 초기화 중...")

        model_path = Path(model_path).resolve()
        if adapter_path:
            adapter_path = Path(adapter_path).resolve()

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            local_files_only=True
        )

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                local_files_only=True
            )

        self.model.eval()
        print("[LLMResponder] 모델 준비 완료")

        self.character_info = (
            "이 캐릭터는 5살 여자아이 요미야. 이름은 요미이고, MBTI는 ESTJ야. "
            "좋아하는 건 치킨, 강아지, 술래잡기, 그림 그리기야. 무서운 건 번개고, 칭찬받으면 기뻐해."
        )

    def wrap_prompt(self, text, emotion=None, event=None, mbti=None, vision=None):
        meta_info = []
        if emotion:
            meta_info.append(f"사용자 감정은 '{emotion}'이야.")
        if event:
            meta_info.append(f"상황은 '{event}'이야.")
        if mbti:
            meta_info.append(f"사용자 MBTI는 '{mbti}'야.")
        if vision:
            meta_info.append(f"시각 정보: {vision}")

        meta_str = " ".join(meta_info)

        return (
            f"### 캐릭터 요약\n{self.character_info}\n\n"
            f"### 사용자 정보\n{meta_str if meta_str else '정보 없음'}\n\n"
            "### 시스템 지침\n"
            "너는 유아야. 입력 내용을 보고 상황에 맞는 유아 말투의 자연스러운 반응을 해줘.\n"
            "그리고 반드시 감정을 함께 추론해. 감정은 다음 8가지 중 하나만 선택해:\n"
            "- 기쁨, 신뢰, 공포, 놀람, 슬픔, 혐오, 분노, 기대\n"
            "감정이 불명확하면 기본값은 '기쁨'이야.\n\n"
            "출력은 아래 형식을 따라야 해:\n"
            "예시:\n"
            "\"대답\": 안녕! 난 요미야!\n"
            "\"감정\": 기쁨\n\n"
            f"### 입력 정보\n{text}\n"
            "### 출력:"
        )

    def generate_response(self, text, emotion=None, event=None, mbti=None):
        prompt = self.wrap_prompt(text, emotion=emotion, event=event, mbti=mbti)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        end = time.time()
        print(f"[LLMResponder] 추론 시간: {end - start:.2f}초")

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        print("[🧪 모델 출력 원문]", repr(output_text))

        # 라인 기반 추출
        response_line = None
        emotion_line = None

        for line in output_text.splitlines():
            if '"대답"' in line:
                response_line = line
            elif '"감정"' in line:
                emotion_line = line

        if response_line and emotion_line:
            response_text = re.sub(r'"?대답"?\s*:\s*["“”]?', '', response_line).strip().strip('"”')
            emotion_text = re.sub(r'"?감정"?\s*:\s*["“”]?', '', emotion_line).strip().strip('"”')

            if "<유아 말투 문장>" in response_text or not response_text:
                return '"대답": 못들었어 다시 말해줘!\n"감정": 기쁨'

            return f'"대답": {response_text}\n"감정": {emotion_text}'

        # 감정만 추출된 경우
        emotion_match = re.search(r'(기쁨|신뢰|공포|놀람|슬픔|혐오|분노|기대)', output_text)
        if emotion_match:
            emotion = emotion_match.group(1)
            response_text = output_text.replace(emotion, "").strip()
            return f'"대답": {response_text}\n"감정": {emotion}'

        # fallback
        print("[⚠️ 경고] 형식 추출 실패. fallback 사용")
        return f'"대답": 못들었어 다시 말해줘!\n"감정": 기쁨'
