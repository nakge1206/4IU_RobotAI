import json

# 경로 설정
input_path = "robot_core/fine_tuning_data/estj_va.jsonl"  # 또는 .json
output_path = "robot_core/fine_tuning_data/estj_va.jsonl"

# 결과 저장 리스트
cleaned_data = []

# 파일 읽기
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        for msg in item["messages"]:
            if msg["role"] == "user":
                # MBTI 정보 제거
                msg["content"] = msg["content"].replace("\nMBTI: ESTJ", "").replace("\nMBTI: INFP", "")
        cleaned_data.append(item)

# 새 파일로 저장
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in cleaned_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print("✅ 완료: MBTI 정보 제거된 cleaned_file.jsonl 생성됨")
