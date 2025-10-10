import os
import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 모델 및 토크나이저 로딩 (CPU용)
model_name = "skt/A.X-4.0-Light"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    trust_remote_code=True
)

# 2. LoRA 설정 (GSQ 방식)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)

# 3. 데이터셋 로딩
dataset = load_dataset("json", data_files="yomi_core/llm_core/new_dataset.json", split="train")

def tokenize_function(example):
    # input dict를 문자열로 변환
    input_str = ""
    if isinstance(example["input"], dict):
        parts = []
        for k, v in example["input"].items():
            parts.append(f"{k}: {v}")
        input_str = " | ".join(parts)
    else:
        input_str = str(example["input"])

    # 최종 프롬프트
    full_prompt = f"""[INSTRUCTION] {example['instruction']} [INPUT] {input_str} [RESPONSE] {example['output']}"""
    
    tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function)

# 4. 학습 설정
output_dir = f"./ax4_lora_finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=True,
    no_cuda=False
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 5. 학습 시작
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
