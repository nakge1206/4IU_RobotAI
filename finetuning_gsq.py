# finetuning_gsq.py - GSQ 튠 전용 학습 코드 (Bllossom 3B Korean 모델 + bitsandbytes 미사용)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import torch
import json

# 1. 모델/토크나이저 불러오기
MODEL_ID = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,      #  bfloat16 → float16
    device_map="auto"
)

# 2. LoRA 구성 (GSQ 방식 어댑터 학습)
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

#  3. 데이터셋 로딩
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

dataset = load_jsonl("gsq_emotion_data.jsonl")

# 4. 프롬프트 포맷 정의
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

#  5. 토크나이징
MAX_LEN = 512
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

tokenized_dataset = formatted_dataset.map(tokenize)

#  6. 학습 설정
training_args = TrainingArguments(
    output_dir="./gsq_finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    fp16=True,                    #  float16 훈련
    bf16=False,                   #  bfloat16 오류, 사용 금지
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_prefetch_factor=None,  # 에러 방지
    dataloader_num_workers=0          # multiprocessing 방지
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 7. 학습 시작
trainer.train()

# 8. LoRA adapter 저장
model.save_pretrained("./gsq_lora_adapter")
tokenizer.save_pretrained("./gsq_lora_adapter")
print("학습 완료 및 어댑터 저장 완료")
