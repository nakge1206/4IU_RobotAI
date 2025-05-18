from openai import OpenAI
import os
from dotenv import load_dotenv  # ← 이 줄 추가

# .env 파일의 내용을 환경변수로 불러옴
load_dotenv()

# API 키 불러오기
api_key = os.getenv("OPENAI_API_KEY")

# 키가 없다면 에러 발생
if not api_key:
    raise ValueError("❌ .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다!")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)

# 파일 업로드 (예시)
client.files.create(
    file=open("robot_core/fine_tuning_data/infp.jsonl", "rb"),
    purpose="fine-tune"
)
