import json

input_path = "robot_core/gsq_emotion_data_va.jsonl"
output_paths = {
    "INFP": "robot_core/fine_tuning_data/infp_va.jsonl",
    "ESTJ": "robot_core/fine_tuning_data/estj_va.jsonl"
}

# MBTI별 저장소
mbti_data = {
    "INFP": [],
    "ESTJ": []
}

# 공통 system 메시지
system_prompt = (
    "너는 7살 유아야. 상황에 따라 감정 표현을 잘 하고, 반말을 쓰며, 친구처럼 말해. "
    "너무 어렵게 말하지 말고, 귀엽고 자연스럽게 이야기해."
)

# 입력 파일 처리
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue  # 빈 줄 무시
        item = json.loads(line)
        mbti = item["input"].replace("MBTI:", "").strip().upper()
        if mbti in mbti_data:
            chat_item = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{item['instruction']}\n{item['input']}"},
                    {"role": "assistant", "content": item["output"]}
                ]
            }
            mbti_data[mbti].append(chat_item)

# MBTI별 JSONL 파일 저장
for mbti, items in mbti_data.items():
    with open(output_paths[mbti], 'w', encoding='utf-8') as f:
        for entry in items:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

print("✅ 변환 완료: infp.jsonl, estj.jsonl")
