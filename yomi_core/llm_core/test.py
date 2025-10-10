import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "skt/A.X-4.0-Light"
adapter_dir = r"C:\Users\COM\Desktop\yomi\4IU_RobotAI\yomi_core\llm_core\ax4_lora_finetune_20251002_154704"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
model.eval()

prompt = "[INSTRUCTION] 너는 유치원에 다니는 5살 어린아이야 [INPUT] 음성: 뭐하고 놀까?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,                 # 먼저 결정론적으로 받기(원하면 True로)
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_first_response(s: str) -> str:
    # 1) [RESPONSE] 이후 텍스트
    m = re.search(r"\[RESPONSE\](.*)", s, flags=re.S)
    if not m:
        return s.strip()
    tail = m.group(1)

    # 2) 다음 태그([INSTRUCTION]|[INPUT]|[RATIONALE]|[TTS]|[STT]|[DETECTION]) 나오기 전까지 자르기
    stop = re.split(r"\s*\[(INSTRUCTION|INPUT|RATIONALE|TTS|STT|DETECTION)\]", tail, maxsplit=1)[0]

    # 3) 양끝 공백/개행 정리
    return stop.strip()

only_voice = extract_first_response(text)
print(only_voice)
