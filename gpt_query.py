from openai import OpenAI
import os

# OpenAI API 키 환경변수에서 불러오기 (미리 설정해 두어야 함)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt(prompt_text: str, system_prompt: str = "당신은 한국어로 응답하는 유용한 AI 어시스턴트입니다.") -> str:
    """
    Whisper로 인식된 문장을 GPT에 입력하고, 응답을 받아 반환

    :param prompt_text: Whisper로부터 전달된 인식된 문장
    :param system_prompt: GPT 역할 지정을 위한 시스템 메시지 (기본값: 한국어 어시스턴트)
    :return: GPT의 응답 텍스트
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"[GPT 오류] {str(e)}"

# 예시 실행 (테스트용)
if __name__ == "__main__":
    test_input = "오늘 날씨 어때?"
    print("[사용자]", test_input)
    gpt_response = ask_gpt(test_input)
    print("[GPT 응답]", gpt_response)
