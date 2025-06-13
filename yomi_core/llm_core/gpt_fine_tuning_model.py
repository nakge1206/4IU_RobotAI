# fine_tuned_gpt.py

import os
import openai
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")


class FineTunedGPTClient:
    def __init__(self, model_id: str = "ft:gpt-4o-2024-08-06:personal::BYBJcaH7"):
        self.model_id = model_id
        self.system_prompt = (
            "너는 7살 유아야. 그리고 너의 mbti는 infp야. "
            "상황에 따라 감정 표현을 잘 하고, 반말을 쓰며, 친구처럼 말해. "
            "너무 어렵게 말하지 말고, 귀엽고 자연스럽게 이야기해."
            "대화에 .을 쓰지마"
            "너는 항상 대화에서 감정을 표현해야 해."
            "감정은 상황에 맞게 자연스럽게 나와야 해 "
            "대화의 시작에 항상 감정 단어를 말해"
            "단, 감정 단어는 기쁨, 슬픔, 분노, 공포, 놀라움, 혐오, 신뢰, 기대로만 적어."
            "감정 단어를 말한 후에는 .찍지말고 바로 / 로 표시하고 그 뒤로 대화를 시작해"
        )


    def map_emotion_to_wheel(self, emotion_input:str) -> str:
        emotion_wheel =  {
            "기쁨": ["기쁨", "행복", "즐거움", "기분 좋은"],
            "슬픔": ["슬픔", "고독", "우울함", "비통"],
            "분노": ["분노", "화남", "짜증", "좌절"],
            "공포": ["공포", "두려움", "불안", "겁"],
            "놀라움": ["놀라움", "충격", "깜짝 놀람", "어리둥절"],
            "혐오": ["혐오", "불쾌", "역겨움", "싫어"],
            "신뢰": ["신뢰", "안도", "믿음", "기대감"],
            "기대": ["기대", "흥분", "설렘", "기대되는"]
        }  
        for key, values in emotion_wheel.items():
            if any(value in emotion_input for value in values):
                    return key
            return "기대" # defult value

    def extract_emotion(self, response: str) -> str:
        base_emotions = ["기쁨", "슬픔", "분노", "공포", "놀라움", "혐오", "신뢰", "기대"]
        prefix = response.split("/")[0].strip()

        for emo in base_emotions:
            if emo in prefix:
                return emo
            
        return "기대"


    def build_instruction(self, stt_text: str, emotion: str, event: str, vision:str=None) -> str:
        stt_text = stt_text.strip()
        emotion = self.map_emotion_to_wheel(emotion)

        if vision:
            return f"청각정보는{stt_text}이고, 감정은 '{emotion}', 상황은 '{event}'이야. \n 시각정보는 {vision}이야."
        else:
            return f"청각정보는 {stt_text}이고, 감정은 '{emotion}', 상황은 '{event}'이야."
        
    
    def build_instruction_vision(self, visionText):
        return f"너의 시각에서 {visionText}가 인식되고 있어."
    

    def chat(self, user_input: str) -> Tuple[str, str]:
        try:
            response = openai.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150
            )
            full_text = response.choices[0].message.content.strip()
            
            emotion_part = full_text.split('/', 1)[0].strip()
            response_text = full_text.split('/', 1)[1].strip()

            emotion = self.extract_emotion(emotion_part)
            return emotion, response_text

        except Exception as e:
            return "오류", f"(GPT 오류: {str(e)})"



if __name__ == "__main__":

        gpt_client = FineTunedGPTClient(model_id="ft:gpt-4o-2024-08-06:personal::BYBJcaH7")
                
        while(True):    
            user_input = input("사용자 입력을 입력하세요: ")  
            emotion, response = gpt_client.chat(user_input)
            print(f"{emotion}")
            print(f"{response}")
