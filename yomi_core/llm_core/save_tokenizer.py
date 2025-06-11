from transformers import AutoTokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join("robot_core", ".env"))
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN이 .env에 없습니다!")

login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("beomi/llama-2-ko-7b", use_fast=True)
tokenizer.save_pretrained("beomi_tokenizer")

print(" Fast tokenizer 저장 완료!")
