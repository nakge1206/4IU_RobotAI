# fine_tuned_gpt.py
import os
import openai

# API 키 로딩
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_instruction(stt_text: str, emotion: str, event: str) -> str:
    stt_text = stt_text.strip()
    return f"{stt_text}... 감정은 '{emotion}'이고 상황은 '{event}'이야."


# 파인튜닝된 GPT 호출 함수 정의
def chat_with_finetuned_gpt(user_input: str) -> str:
    response = openai.chat.completions.create(

        model="ft:gpt-4o-2024-08-06:personal::BYBJcaH7",  # 사용자 모델 ID

        messages=[
            {
                "role": "system",
                "content": (
                    "너는 7살 유아야. 그리고 너의 mbti는 infp야. "
                    "상황에 따라 감정 표현을 잘 하고, 반말을 쓰며, 친구처럼 말해. "
                    "너무 어렵게 말하지 말고, 귀엽고 자연스럽게 이야기해."
                )
            },
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )

    return response.choices[0].message.content
