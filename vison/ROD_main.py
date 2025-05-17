import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import cv2
from time import time
from ROD_detection import RealtimeObjectDetection
from ROD_log import DetectionLogger

def run_detection():
    detector = RealtimeObjectDetection()
    logger = DetectionLogger()

    cap = cv2.VideoCapture(0) # 기본 인지 셋 찾기 0은 카메라
    assert cap.isOpened(), "웹캠을 열 수 없습니다."

    while True: 
        start_time = time()
        ret, frame = cap.read() # 프레임 단위로 읽기
        if not ret:
            break

        detections = detector.ObjectInfomation(frame) # 프레임 단위 감지
        logger.add(detections)

        frame = detector.plot_boxes(detections, frame) # 화면에 프레임 표시
        end_time = time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('YOLOv5 Webcam Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.save()

if __name__ == "__main__":
    run_detection()
