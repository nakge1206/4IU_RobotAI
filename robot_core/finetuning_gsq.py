# finetuning_gsq.py - GSQ 튠 전용 학습 코드 (환경변수 방식)

import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #추가

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from dotenv import load_dotenv  #  .env 파일에서 환경변수 불러오기

#  1. 환경변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()  # .env 파일이 있으면 로드
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("환경변수 HF_TOKEN이 설정되지 않았습니다.")
login(hf_token)

#  2. 모델/토크나이저 불러오기
MODEL_ID = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

#  3. LoRA 구성 (GSQ 방식)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config, adapter_name="default")

#  4. 데이터셋 로딩
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

dataset = load_jsonl("robot_core/gsq_emotion_data.jsonl")

#  5. 프롬프트 포맷 정의
PROMPT_TEMPLATE = """{instruction}

### 사용자 ({input})
{instruction}

### 로봇 (유아 역할)
{output}"""

def format_example(example):
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"]
        )
    }

formatted_dataset = dataset.map(format_example)

#  6. 토크나이징
MAX_LEN = 512
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

tokenized_dataset = formatted_dataset.map(tokenize)

#  7. 학습 설정
training_args = TrainingArguments(
    output_dir="./gsq_finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_prefetch_factor=None,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

#  8. 학습 시작
trainer.train()

#  9. LoRA adapter 저장
model.save_pretrained("./gsq_lora_adapter")
tokenizer.save_pretrained("./gsq_lora_adapter")
print("학습 완료 및 어댑터 저장 완료")