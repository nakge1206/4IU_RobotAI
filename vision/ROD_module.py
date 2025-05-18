import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지 설정
import cv2
import threading
from time import time
from ROD_detection import RealtimeObjectDetection
from ROD_log import DetectionLogger

class YoloModule:
    def __init__(self, interval=1.0, isLog=False, on_vision_callback=None, viewGUI=True):
        self.isLog = isLog
        self.interval = interval #몇초에 한번씩 탐지된 객체를 출력하는지
        self.on_vision_callback = on_vision_callback
        self.viewGUI = viewGUI

        self.detector = RealtimeObjectDetection()
        self.logger = DetectionLogger() if isLog else None

        self.latest_frame = None
        self.detections = None
        self.running = False
        self.recording_thread = None
        self.fps = None

    def run_detection(self):
        self.running = True
        threading.Thread(target=self._detection_worker, daemon=True).start()
        #창은 스레드 분리해서 돌리면 무한 응답없음에 빠짐
        # yolo 창 출력
        try:
            if self.viewGUI:
                while self.running:
                    if self.detections is None or self.latest_frame is None:
                        continue
                    frame = self.detector.plot_boxes(self.detections, self.latest_frame) # 화면에 프레임 표시
                    cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('YOLOv5 Webcam Detection', frame)

                    # 원래 종료는 q를 눌러서임.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cv2.destroyAllWindows()
        

    def stop(self):
        self.running = False
    
    def getDetecions(self):
        return self.detections
    
    def _detection_worker(self):
        cap = cv2.VideoCapture(0) # 기본 인지 셋 찾기 0은 카메라
        assert cap.isOpened(), "웹캠을 열 수 없습니다."

        last_detection_time = 0
        try:
            while self.running: 
                start = time()
                ret, frame = cap.read() # 프레임 단위로 읽기
                self.latest_frame = frame
                if not ret:
                    break

                #cv2 창띄우기 용 detecions
                
                self.detections = self.detector.ObjectInfomation(frame)
                end = time()
                self.fps = 1 / (end - start) if end-start != 0 else 0

                current_time = time()
                if current_time - last_detection_time >= self.interval:
                    last_detection_time = current_time
                    if self.on_vision_callback:
                        self.on_vision_callback(self.detections)
                    if self.logger:
                        self.logger.add(self.detections)


                
        finally:
            cap.release()
            if self.logger:
                self.logger.save()

def handle_detections(dets):
    print("감지된 객체:", dets)

if __name__ == "__main__":
    yolo = YoloModule(isLog=False, interval=2.0, on_vision_callback=handle_detections, viewGUI=False)
    try:
        yolo.run_detection()
        while True:
            pass
    except KeyboardInterrupt:
        print("키보드 인터럽트")
        yolo.stop()
    
