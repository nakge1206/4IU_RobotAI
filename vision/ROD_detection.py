import torch
import cv2
import numpy as np
from datetime import datetime
from time import time

class RealtimeObjectDetection:
    def __init__(self): # 초기화 함수
        self.model = self.load_model() # 학습 모델 다운
        self.classes = self.model.names # 객체 이름 변환
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 사용할 장치 선택 cuda면 cuda 안 되면 cpu로
        self.all_detections = []  # 감지 객체 리스트 변환
        print(f"Using device: {self.device}") 

    def load_model(self): # 학습된 데이터 셋 불러오는 함수
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # yolov5에 어떤 dataset을 쓸건지
        ## model = torch.hub.load( ## 학습된 커스텀 파일 경로
        ##     'ultralytics/yolov5',
        ##     'custom',
        ##     path='C:/yolov5/runs/train/coco_plus_custom/weights/best.pt'  
        ## )
        return model

    def ObjectInfomation(self, frame): # 객체 정보 저장 함수
        self.model.to(self.device) # 모델을 장치로 옮겨서 실행 cuda or cpu
        frame = [frame] # YOLO는 입력 이미지도 리스트로 받아서 frame을 리스트로 받아야함
        results = self.model(frame) # 감지 수행
        labels = results.xyxyn[0][:, -1].cpu().numpy() # [:, -1]로 class_id만 받아옴
        cords = results.xyxyn[0][:, :-1].cpu().numpy() # [:, :-1]로 나머지를 다 받아옴

        x_shape, y_shape = frame[0].shape[1], frame[0].shape[0]
        detections = []

        for i in range(len(labels)):
            row = cords[i] 
            if row[4] >= 0.3:  # 감지 적중률 0.3이상만 감지
                x1 = int(row[0] * x_shape) # 좌표 원래 이미지 크기로 변환
                y1 = int(row[1] * y_shape)
                x2 = int(row[2] * x_shape)
                y2 = int(row[3] * y_shape)
                class_name = self.class_to_label(labels[i]) #객체 이름 변환
                detections.append({ # 리스트에 단일 객체 감지 결과 저장
                    'label': class_name,
                    'confidence': float(row[4]),
                    'box': [x1, y1, x2, y2]
                })

        return detections

    def class_to_label(self, x): # 클래스 번호를 문자열로 바꾸는 함수
        return self.classes[int(x)] 

    def plot_boxes(self, detections, frame): # 감지 객체 박스 치는 함수
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame
