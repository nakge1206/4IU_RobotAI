import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
from llm_inference import LLMResponder

llm = LLMResponder()

texts = ["나랑 같이 놀자", "나 간식이 없어서 슬퍼", "너 왜그렇게 생겼어?", "오늘 너무 기분좋은 날인거 같아", "뭐하고 놀까?"]
for i  in range(5):
    response = llm.generate_response(
        texts[i],
        emotion="",          
        event="",
        mbti="INFP"
    )
    print(f"{i}번째 질문:", texts[i])
    print(f"{i}번째 로봇 응답:", response)