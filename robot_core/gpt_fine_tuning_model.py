# fine_tuned_gpt.py

import os
import openai

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")


class FineTunedGPTClient:
    def __init__(self, model_id: str = "ft:gpt-4o-2024-08-06:personal::BYBJcaH7"):
        self.model_id = model_id
        self.system_prompt = (
            "너는 7살 유아야. 그리고 너의 mbti는 infp야. "
            "상황에 따라 감정 표현을 잘 하고, 반말을 쓰며, 친구처럼 말해. "
            "너무 어렵게 말하지 말고, 귀엽고 자연스럽게 이야기해."
        )

    def build_instruction(self, stt_text: str, emotion: str, event: str, vision:str=None) -> str:
        stt_text = stt_text.strip()
        if vision:
            return f"청각정보는{stt_text}이고, 감정은 '{emotion}', 상황은 '{event}'이야. \n 시각정보는 {vision}이야."
        else:
            return f"청각정보는 {stt_text}이고, 감정은 '{emotion}', 상황은 '{event}'이야."
    
    def build_instruction_vision(self, visionText):
        return f"너의 시각에서 {visionText}가 인식되고 있어."

    def chat(self, user_input: str) -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(GPT 오류: {str(e)})"
