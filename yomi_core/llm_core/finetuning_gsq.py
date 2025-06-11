# finetuning_gsq.py - Beomi/LLaMA2-Ko-7B + GSQ LoRA 튠 점검 포함 버전

import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# 1. 환경변수 로딩
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("환경변수 HF_TOKEN이 설정되지 않았습니다.")
login(hf_token)

# 2. 모델 및 토크나이저
MODEL_ID = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",  # 자동 분배
    low_cpu_mem_usage=True
)

# 3. LoRA 설정 및 GPU 강제 할당
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
model = get_peft_model(model, lora_config)
model = model.to("cuda")  #  GPU 강제 할당

# 디바이스 확인
print("모델 디바이스:", next(model.parameters()).device)

# 4. 데이터셋 로딩 및 포맷
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return Dataset.from_list([json.loads(line) for line in f])

dataset = load_jsonl("robot_core/gsq_emotion_data.jsonl")

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

# 5. 토크나이징
MAX_LEN = 512
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

tokenized_dataset = formatted_dataset.map(tokenize)

# 샘플 데이터 출력 확인
print(" 토크나이즈 예시:", tokenized_dataset[0])
print(" 전체 샘플 수:", len(tokenized_dataset))

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./gsq_lora_beomi7b_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=1,  # 즉시 로그 출력
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 7. 학습 시작 (에러 감지용 try-except)
if __name__ == "__main__":
    try:
        print("학습 시작")
        trainer.train()
        print(" 학습 종료")
        model.save_pretrained("./gsq_lora_adapter")
        tokenizer.save_pretrained("./gsq_lora_adapter")
        print(" 어댑터 저장 완료")
    except Exception as e:
        print(f" 예외 발생: {e}")
