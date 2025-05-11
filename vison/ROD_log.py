import json
from datetime import datetime

class DetectionLogger:
    def __init__(self):
        self.log = []

    def add(self, detections):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 타임 로그 저장
        self.log.append({ # 리스트에 전체 프레임 감지 결과 저장
            'time': timestamp,
            'detections': detections
        })

    def save(self, filename='detections_log.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)
        print(f"감지 로그가 '{filename}' 파일로 저장되었습니다.")
