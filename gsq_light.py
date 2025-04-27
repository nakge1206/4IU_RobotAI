# gsq_light.py - TinyLlama 1B용, 최소화 GSQ 문장 생성 테스트

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

def quantize_tensor(tensor, group_size=32):
    shape = tensor.shape
    flat = tensor.flatten()
    n = len(flat)
    num_groups = math.ceil(n / group_size)
    scales = []
    q_values = []
    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, n)
        group = flat[start:end]
        scale = group.abs().max() / 31
        scale = scale if scale != 0 else 1e-6
        q = torch.clamp((group / scale).round(), -31, 31).to(torch.int8)
        scales.append(scale)
        q_values.append(q)
    return torch.cat(q_values), torch.tensor(scales), shape

def dequantize_tensor(q_tensor, scales, shape, group_size=32):
    n = q_tensor.numel()
    output = torch.zeros(n, dtype=torch.float32)
    num_groups = math.ceil(n / group_size)
    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, n)
        output[start:end] = q_tensor[start:end].float() * scales[i]
    return output.view(shape)

def int_matmul(q_weight, weight_scales, q_input, input_scales, shape_w, shape_in):
    deq_w = dequantize_tensor(q_weight, weight_scales, shape_w)
    deq_x = dequantize_tensor(q_input, input_scales, shape_in)
    return torch.matmul(deq_x, deq_w.T)

# TinyLlama 사용
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("[로딩] 토크나이저 불러오는 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
print("[로딩] LLM 모델 불러오는 중...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()

# 설정
max_steps = 3  # 3개 토큰 생성
vocab_size = 256  # 256개만 softmax

prompt = "슬퍼 보이는 친구에게 어떻게 말해줘야 할까?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :5].to(model.device)

# lm_head weight 양자화
hidden_size = model.config.hidden_size
with torch.no_grad():
    lm_weight = model.lm_head.weight[:vocab_size, :hidden_size].detach().to(torch.float32)
    q_w, s_w, shape_w = quantize_tensor(lm_weight)

print("[GSQ 문장 생성 시작]")
for step in range(max_steps):
    with torch.no_grad():
        cur_id = input_ids[:, -1:]
        embed = model.model.embed_tokens(cur_id).detach().to(torch.float32)
        q_in, s_in, shape_in = quantize_tensor(embed)

        out = int_matmul(q_w, s_w, q_in, s_in, shape_w, shape_in)
        probs = F.softmax(out[:, -1], dim=-1)
        next_id = torch.argmax(probs, dim=-1, keepdim=True).to(model.device)
        input_ids = torch.cat([input_ids, next_id], dim=-1)

# 출력
print("[GSQ 응답]", tokenizer.decode(input_ids[0], skip_special_tokens=True))
