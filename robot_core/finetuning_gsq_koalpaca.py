import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# 1. 모델 및 토크나이저 로드
model_name = "beomi/KoAlpaca-Polyglot-12.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None  #  자동 offloading 방지
).to("cuda")  #  명시적 GPU 이동

# 2. LoRA 설정 (GSQ 스타일 반영 가능)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]  # GPT-NeoX 기반 구조
)
model = get_peft_model(model, peft_config)

# 3. 데이터셋 로딩 (예시: Hugging Face의 Alpaca 형식)
dataset = load_dataset("yahma/alpaca-cleaned")  # 'instruction', 'input', 'output' 필드 포함

# 4. 전처리 함수 정의
def preprocess(example):
    prompt = f"{example['instruction']}\n{example['input']}" if example["input"] else example["instruction"]
    prompt += "\n\n### 답변:\n" + example["output"]
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 5. 전처리 적용
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# 6. 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./gsq_lora_adapter_koalpaca",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# 7. Trainer 초기화 및 학습
model.config.use_cache = False  # LoRA 학습 시 필수 설정

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

# 8. 저장
model.save_pretrained("./gsq_lora_adapter_koalpaca")
tokenizer.save_pretrained("./gsq_lora_adapter_koalpaca")