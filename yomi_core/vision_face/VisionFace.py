import os
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QTransform, QMovie

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import threading
import time
from ROD_detection import RealtimeObjectDetection
from ROD_log import DetectionLogger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지

class YomiFace(QWidget):
    def __init__(self):
        super().__init__()
        # self.setWindowFlags(Qt.FramelessWindowHint) #혹시 있는 여백 투명색으로 해줌
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.init_ui()

    def init_ui(self):
        self.showFullScreen()

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)  # 자동 사이즈 조정

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def set_emotion_pixmap(self, pixmap: QPixmap):
        self.label.setPixmap(pixmap)
        
    def set_emotion_movie(self, gif_path: str):
        if hasattr(self, 'movie') and self.movie is not None:
            self.label.clear()
            self.movie.stop()
            del self.movie

        self.movie = QMovie(gif_path)
        if not self.movie.isValid():
            print(f"[Face_ERROR] GIF 로딩 실패: {gif_path}")
            return

        from PyQt5.QtCore import QSize
        self.movie.setScaledSize(QSize(self.width(), self.height()))
        self.label.setMovie(self.movie)
        self.movie.start()

    def resizeEvent(self, event):
        if hasattr(self, 'movie') and self.movie is not None:
            self.movie.setScaledSize(self.size())
        super().resizeEvent(event)



class FaceController(QObject):
    def __init__(self):
        super().__init__()
        self.gui = None  # GUI 객체를 받아서 제어

        self.emotions = ["joy", "sadness", "angry", "fear", "surprise", "disgust", "trust", "anticipation"]
        self.emotion_index = 0
        self.blinking = False
        self.img_index = 0

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_paths = self._load_image_paths()
        self.scaled_cache = {}

    def initialize(self, gui):
        self.gui = gui
        self.gui.set_emotion_movie(self.image_paths)
        # self.timer = QTimer()
        # self.timer.setInterval(100)
        # self.timer.timeout.connect(self.update_emotion)
        # self.timer.start()

    def _load_image_paths(self):
        """감정에 맞는 이미지 경로 찾기"""
        base = os.path.join(self.script_dir, 'blackface')
        return os.path.join(base, 'yomi_1.gif')
        # return {
        #     emotion: (
        #         os.path.join(base, f'black_{emotion}D.png'),
        #         os.path.join(base, f'black_{emotion}S.png')
        #     ) for emotion in self.emotions
        # }

    # def update_emotion(self):
    #     """현재 감정에 따라 GUI업데이트"""
    #     if self.blinking:
    #         self.img_index = (self.img_index + 1) % 2
    #     else:
    #         self.img_index = 0

    #     emotion = self.emotions[self.emotion_index]
    #     cache_key = (emotion, self.img_index)

    #     if cache_key in self.scaled_cache:
    #         pixmap = self.scaled_cache[cache_key]
    #     else:
    #         path = self.image_paths[emotion][self.img_index]
    #         path = self.image_paths
    #         pixmap = QPixmap(path)
    #         if pixmap.isNull():
    #             print(f"[Face_ERROR] 이미지 불러오기 실패 : {path}")
    #             return
    #         pixmap = pixmap.scaled(
    #             self.gui.width(), self.gui.height(),
    #             Qt.IgnoreAspectRatio, Qt.SmoothTransformation
    #         )
    #         self.scaled_cache[cache_key] = pixmap

    #     self.gui.set_emotion_pixmap(pixmap)

    def update_emotion(self):
        """단일 GIF 애니메이션 디버깅"""
        pass

    def set_blinking(self, value: bool):
        """입 움직이는지 설정"""
        self.blinking = value

    def next_emotion(self):
        """디버그용도 : space누르면 감정 바뀜"""
        self.emotion_index = (self.emotion_index + 1) % len(self.emotions)
        print(f"[Face_Controller] 감정 설정 : {self.emotions[self.emotion_index]}")

    def set_emotion(self, emotion: str):
        """감정 이름으로 직접 감정을 설정"""
        if emotion in self.emotions:
            self.emotion_index = self.emotions.index(emotion)
            print(f"[Face_Controller] 감정 설정 : '{emotion}'")
        else:
            print(f"[Face_WARN] Unknown emotion: '{emotion}'")
 
    def stop(self):
        #self.timer.stop()
        print("[Face_Controller] Stopped")

class YoloWorker(threading.Thread):
    def __init__(self, interval=1.0, isLog=False, on_vision_callback=None):
        super().__init__()
        self.detector = RealtimeObjectDetection()
        self.logger = DetectionLogger() if isLog else None
        self.on_vision_callback = on_vision_callback
        self.interval = interval

        self.frame = None
        self.detections = None
        self.running = True
        self.lock = threading.Lock()
        self.prev_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[YOLO] 카메라 열기 실패")
            return
        
        last_detection_time = 0
        while self.running:
            ret, frame = cap.read()
            #print(frame.shape)
            if not ret:
                continue
            detection = self.detector.ObjectInfomation(frame)
            with self.lock:
                self.frame = frame.copy()
                self.detections = detection
            time.sleep(0.01)

            current_time = time.time()
            if current_time - last_detection_time >= self.interval:
                last_detection_time = current_time
                if self.on_vision_callback:
                    self.on_vision_callback(self.detections)
                if self.logger:
                    self.logger.add(self.detections)

        cap.release()
        if self.logger:
            self.logger.save()
            print("[YOLO] 로그 저장 완료")

    def stop(self):
        self.running = False
        self.join()

    def _get_latest(self):
        """내부 GUI용 이미지, detection 반환 함수"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.detections
    
    def get_detections(self):
        """외부에서 감지 결과만 받아갈 때 사용"""
        with self.lock:
            return self.detections.copy() if self.detections else []

class VisonFaceMain:
    def __init__(self, interval=1.0, isLog=False, on_vision_callback=None, viewGUI=True, isFPS=False):
        self.app = QApplication(sys.argv)

        # Face UI 및 제어
        self.face = YomiFace()
        self.face.show()

        self.controller_thread = QThread()
        self.controller = FaceController()
        self.controller.moveToThread(self.controller_thread)
        self.controller_thread.started.connect(lambda: self.controller.initialize(self.face))
        self.controller_thread.start()

        # YOLO 감지 스레드
        self.yoloWorker = YoloWorker(interval=interval, isLog=isLog, on_vision_callback=on_vision_callback)
        self.yoloWorker.start()

        self.viewGUI = viewGUI
        self.running = True
        self.isFPS = isFPS

        self.face.keyPressEvent = self.key_handler
        self.app.aboutToQuit.connect(self._quit)

    def key_handler(self, event):
        """
        space - 다음 감정 변환
        B - 입 뻥긋 활성화
        Y - 입 뻥긋 비활성화
        Q - 종료
        """
        if event.key() == Qt.Key_Space:
            self.controller.next_emotion()
        elif event.key() == Qt.Key_B:
            self.controller.set_blinking(True)
        elif event.key() == Qt.Key_Y:
            self.controller.set_blinking(False)
        elif event.key() == Qt.Key_Q:
            self.running = False
            self.app.quit()

    def _quit(self):
        """내부 종료문"""
        self.yoloWorker.stop()
        self.controller.stop()
        self.controller_thread.quit()
        self.controller_thread.wait()
        if self.viewGUI:
            cv2.destroyAllWindows()

    def stop(self):
        """VisionFace 전체 종료"""
        self.running = False
        self.app.quit()

    def vision_get_detections(self):
        """현재 탐지한 객체 받아오기"""
        return self.yoloWorker.detections

    def face_set_blinking(self, value: bool):
        """입 움직이는지 설정"""
        self.controller.set_blinking(value)

    def face_set_emotion(self, emotion: str):
        """감정 변경"""
        self.controller.set_emotion(emotion)

    def run(self):
        """실행"""
        while self.running:
            frame, detections = self.yoloWorker._get_latest()
            if frame is not None:
                if self.viewGUI:
                    # print(detections)
                    frame = self.yoloWorker.detector.plot_boxes(detections, frame)
                    frame = cv2.flip(frame, -1)

                    if self.isFPS:
                        curr_time = time.time()
                        fps = 1.0 / (curr_time - self.yoloWorker.prev_time)
                        self.yoloWorker.prev_time = curr_time

                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("YOLO", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
            
            #PyQt 실행 함수
            self.app.processEvents()

        self._quit()




if __name__ == '__main__':
    main = VisonFaceMain()
    try:
        main.run()
    except KeyboardInterrupt:
        main.stop()
        sys.exit(0)

    
