import os
import cv2
import threading
import time
from ROD_detection import RealtimeObjectDetection
from ROD_log import DetectionLogger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지

class YoloModule:
    def __init__(self, interval=1.0, isLog=False, on_vision_callback=None, viewGUI=True):
        self.isLog = isLog
        self.interval = interval
        self.on_vision_callback = on_vision_callback
        self.viewGUI = viewGUI

        self.detector = RealtimeObjectDetection()
        self.logger = DetectionLogger() if isLog else None

        self.latest_frame = None
        self.detections = None
        self.running = False
        self.fps = 0.0

    def start(self):
        self.running = True
        threading.Thread(target=self._detection_worker, daemon=True).start()

        if self.viewGUI:
            self._display_worker()

    def stop(self):
        self.running = False

    def getDetections(self):
        return self.detections

    def _detection_worker(self):
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "웹캠을 열 수 없습니다."

        last_detection_time = 0
        try:
            while self.running:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 프레임 읽기 실패")
                    continue

                self.latest_frame = frame
                self.detections = self.detector.ObjectInfomation(frame)

                end = time.time()
                self.fps = 1 / (end - start) if end - start > 0 else 0

                current_time = time.time()
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

    def _display_worker(self):
        timeout = time.time() + 5  # 5초 기다림
        while self.running and (self.latest_frame is None or self.detections is None):
            if time.time() > timeout:
                print("⚠️ 초기화 지연: 5초 내에 frame/detection이 없음")
                return
            time.sleep(0.1)

        while self.running:
            try:
                if self.latest_frame is None or self.detections is None:
                    print("⚠️ 아직 프레임 또는 감지 결과가 준비되지 않음")
                    continue
                frame = self.detector.plot_boxes(self.detections, self.latest_frame.copy())
                frame = cv2.flip(frame, 0)
            except Exception as e:
                print("🛑 plot_boxes 실패:", e)
                continue

            cv2.putText(frame, f"FPS: {int(self.fps)}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('YOLOv5 Webcam Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()



def handle_detections(dets):
    print("감지된 객체:", dets)

if __name__ == "__main__":
    yolo = YoloModule(isLog=False, interval=2.0, on_vision_callback=handle_detections, viewGUI=True)
    try:
        yolo.start()
        while yolo.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C 감지: 종료 중...")
        yolo.stop()
