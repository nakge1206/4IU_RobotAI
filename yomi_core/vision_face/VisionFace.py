# 기본 패키지
import os
import sys
import threading
import time

# Face 패키지
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QMovie
import rospy
from std_msgs.msg import Bool

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Vision 패키지
import cv2
from ROD_detection import RealtimeObjectDetection
from ROD_log import DetectionLogger

import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 방지

import pathlib # 이것부터 5줄을 통해서 마치 WindowsPath가 있는것처럼 꾸미기 <- 어짜피 실제로 기능하는데는 문제가 없기 때문
from pathlib import PosixPath

# 🔧 WindowsPath를 PosixPath로 대체 (Ubuntu에서도 언피클 가능)
class WindowsPath(PosixPath):
    """Fake WindowsPath for loading Windows-trained models on Linux"""
    pass
pathlib.WindowsPath = WindowsPath

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
            print(f"[VisionFace] [YomiFace] GIF 로딩 실패: {gif_path}")
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
    emotion_changed = pyqtSignal(str)
    def __init__(self, mbti = "I"):
        super().__init__()
        self.gui = None  # GUI 객체를 받아서 제어
        self.mbti = mbti

        self.emotions = ["joy", "sadness", "angry", "fear", "surprise", "disgust", "trust", "anticipation", "no"]
        self.emotion_index = 0

        self.emotion_changed.connect(self.set_emotion)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_paths = self._load_image_paths()

        self.tts_state = False
        self.no_face_trigger_time = None
        if not rospy.core.is_initialized():
            rospy.init_node('vision_face_node', anonymous=True)
        rospy.Subscriber("/tts_state", Bool, self.handle_tts)
        

    def initialize(self, gui):
        self.gui = gui
        current_emotion = self.emotions[self.emotion_index]

        self.gui.set_emotion_movie(self.image_paths[current_emotion])

    def _load_image_paths(self):
        """감정에 맞는 이미지 경로 찾기"""
        base = os.path.join(self.script_dir, 'face')
        paths = {}
        for emotion in self.emotions:
            gif_path = os.path.join(base, f"{self.mbti}_{emotion}.gif")
            if not os.path.exists(gif_path):
                print(f"[VisionFace] [FaceController] (_load_image_paths) 경고 : {gif_path} 파일 없음")
            paths[emotion] = gif_path
        return paths
    
    def update_emotion(self):
        """현재 감정에 따라 GUI업데이트"""
        current_emotion = self.emotions[self.emotion_index]
        gif_path = self.image_paths.get(current_emotion)
        if gif_path and os.path.exists(gif_path):
            self.gui.set_emotion_movie(gif_path)
        else:
            print(f"[VisionFace] [FaceController] (update_emotion) GIF 없음: {gif_path}")

    def set_emotion(self, emotion: str):
        """감정 이름으로 직접 감정을 설정"""
        if emotion in self.emotions:
            self.emotion_index = self.emotions.index(emotion)
            print(f"[VisionFace] [FaceController] (set_emotion) 감정 설정 : '{emotion}'")
        else:
            print(f"[VisionFace] [FaceController] (set_emotion) Unknown emotion: '{emotion}'")
        self.update_emotion()
    
    def handle_tts(self, msg):
        """rostopic(stt_state) 감지해서 True->False일때 실행"""
        # print(f"[VisionFace] [FaceController] (handle_tts) tts변화 감지함 : {msg.data}")
        if self.tts_state and not msg.data:
            self.no_face_trigger_time = time.time() + 6
        self.tts_state = msg.data

    def check_no_face(self):
        """무표정트리커 활성화 되면, n초후 출력"""
        if self.no_face_trigger_time and time.time() >= self.no_face_trigger_time:
            self.no_face_trigger_time = None
            # gif_path = os.path.join(self.script_dir, 'face', f"{self.mbti}_no.gif")
            # if os.path.exists(gif_path) and self.gui:
            #     self.gui.set_emotion_movie(gif_path)
            if self.gui:
                self.emotion_index = self.emotions.index("no")
                self.update_emotion()
                print("[VisionFace] [FaceController] (check_no_face) 감정 설정 : 무표정")
    
    # def _delayed_no_face(self):
    #     """TTS가 완료되고 나서, 3초 후 무표정 실행"""
    #     gif_path = os.path.join(self.script_dir, 'face', f"{self.mbti}_no.gif")
    #     if os.path.exists(gif_path) and self.gui:
    #         self.gui.set_emotion_movie(gif_path)
    #         self.update_emotion()
    #         print(f"[VisionFace] [FaceController] (handle_tts) 감정 설정 : 무표정")
    #     else:
    #         print(f"[VisionFace] [FaceController] 무표정 파일 없음: {gif_path}")

    def next_emotion(self):
        """디버그용 : 
            space누르면 감정 바뀜"""
        self.emotion_index = (self.emotion_index + 1) % len(self.emotions)
        self.update_emotion()
        print(f"[VisionFace] [FaceController] (next_emotion) 감정 설정 : {self.emotions[self.emotion_index]}")
    
    def next_mbti(self):
        """디버그용 : 
            M누르면 mbti 바뀜"""
        self.mbti = "E" if self.mbti == "I" else "I"
        self.image_paths = self._load_image_paths()
        self.update_emotion()
        print(f"[VisionFace] [FaceController] (next_mbti) mbti 설정 : {self.emotions[self.emotion_index]}")


class YoloWorker(threading.Thread):
    def __init__(self, interval=1.0, isLog=False, on_vision_callback=None):
        super().__init__()
        self.detector = RealtimeObjectDetection()
        self.logger = DetectionLogger() if isLog else None
        self.on_vision_callback = on_vision_callback
        self.interval = interval

        self.frame = None
        self.detectInfo = None
        self.countInfo=None
        self.running = True
        self.lock = threading.Lock()
        self.prev_time = time.time()

    def run(self):
        cap = None
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
        if cap.isOpened():
            print(f"[VisionFace] [YoloWorker] 카메라 {i} 열기 성공")
        else:
            print("[VisionFace] [YoloWorker] 카메라 열기 실패")
            return
        
        last_detection_time = 0
        while self.running:
            ret, frame = cap.read()
            #print(frame.shape)
            if not ret: # 이미지가 제대로 읽히지 않을때
                print("[VisionFace] [YoloWorker] 프레임 읽기 실패. 카메라를 재연결 시도합니다.")
                cap.release()  # 이전 카메라 객체를 종료
                for i in range(6):  # 0부터 5까지 다시 시도
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        print(f"[VisionFace] [YoloWorker] 카메라 {i} 열기 성공")
                        break
                continue
            #이미지 반전
            # frame = cv2.flip(frame, -1)

            detections, class_counts = self.detector.ObjectInfomation(frame)
            with self.lock:
                self.frame = frame.copy()
                self.detectInfo = detections
                self.countInfo = class_counts
            time.sleep(0.01)

            current_time = time.time()
            if current_time - last_detection_time >= self.interval:
                last_detection_time = current_time
                if self.on_vision_callback:
                    self.on_vision_callback(self.detectInfo, self.countInfo)
                if self.logger:
                    self.logger.add(self.detectInfo, self.countInfo)

        cap.release()
        if self.logger:
            self.logger.save()
            print("[VisionFace] [YoloWorker] 로그 저장 완료")

    def stop(self):
        print("[VisionFace] [YoloWorker] YoloWorker 종료")
        self.running = False
        self.join()

    def _get_latest(self):
        """내부 GUI용 이미지, detection 반환 함수"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.detectInfo
    
    def get_detections(self):
        """외부에서 감지 결과만 받아갈 때 사용"""
        with self.lock:
            return self.detectInfo.copy() if self.detectInfo else []

class VisonFaceMain:
    def __init__(self, interval=1.0, isLog=False, on_vision_callback=None, viewGUI=True, isFPS=False, mbti="I"):
        self.app = QApplication(sys.argv)

        # Face UI 및 제어
        self.face = YomiFace()
        self.face.show()

        self.controller_thread = QThread()
        self.controller = FaceController(mbti)
        self.controller.initialize(self.face)
        self.controller.moveToThread(self.controller_thread)
        self.controller_thread.started.connect(lambda: self.controller.initialize(self.face))
        self.controller_thread.start()

        # YOLO 감지 스레드
        self.yoloWorker = YoloWorker(interval=interval, isLog=isLog, on_vision_callback=on_vision_callback)
        self.yoloWorker.start()

        self.viewGUI = viewGUI
        self.running = True
        self.isFPS = isFPS

        self.startTimer = None
        self.stopTimer = None
        self.currTimer = None

        self.face.keyPressEvent = self.key_handler
        self.app.aboutToQuit.connect(self._quit)

    def key_handler(self, event):
        """
        디버그용 : 
            space - 다음 감정 변환
            M - E <-> I 변경
            Q - 종료
            """
        if event.key() == Qt.Key_Space:
            self.controller.next_emotion()
        elif event.key() == Qt.Key_M:
            self.controller.next_mbti()
        elif event.key() == Qt.Key_D:
            self.startTimer = time.time()
            detect = self.vision_get_detections()
            self.stopTimer = time.time()
            print(f"객체 : {detect}")
            print(f"지연율 : {self.stopTimer-self.startTimer}")
        elif event.key() == Qt.Key_F:
            emotions = ["joy", "sadness", "angry", "fear", "surprise", "disgust", "trust", "anticipation", "no"]
            emotion_index = random.randrange(0,8)
            self.startTimer = time.time()
            self.face_set_emotion(emotions[emotion_index])
            self.stopTimer = time.time()
            print(f"지연율 : {self.stopTimer-self.startTimer}")
        elif event.key() == Qt.Key_Q:
            self.running = False
            self.app.quit()

    def _quit(self):
        """내부 종료문"""
        self.yoloWorker.stop()
        self.controller_thread.quit()
        self.controller_thread.wait()
        if self.viewGUI:
            cv2.destroyAllWindows()

    def stop(self):
        """VisionFace 전체 종료"""
        print("[VisionFace] [VisionFaceMain] 모듈 종료")
        self.running = False
        self.app.quit()

    def vision_get_detections(self):
        """현재 탐지한 객체 받아오기"""
        return self.yoloWorker.detections

    def face_set_emotion(self, emotion: str):
        """감정 변경"""
        self.controller.emotion_changed.emit(emotion)
        # self.controller.set_emotion(emotion)

    def run(self):
        """실행"""
        while self.running:
            frame, detections = self.yoloWorker._get_latest()
            if frame is not None:
                if self.viewGUI:
                    # print(detections)
                    frame = self.yoloWorker.detector.plot_boxes(detections, frame)
                    #카메라 뒤집기
                    # frame = cv2.flip(frame, -1)

                    if self.isFPS:
                        curr_time = time.time()
                        fps = 1.0 / (curr_time - self.yoloWorker.prev_time)
                        self.yoloWorker.prev_time = curr_time

                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("YOLO", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
            if detections:
                print("[VisionFace] 감지:", self.yoloWorker.countInfo)
            
            #PyQt 실행 함수
            self.controller.check_no_face()
            self.app.processEvents()

        self._quit()



if __name__ == '__main__':
    start_time = time.time()
    main = VisonFaceMain(mbti="I")
    try:
        main.run()
    except KeyboardInterrupt:
        main.stop()
        sys.exit(0)
    end_time = time.time()
    running_time = end_time - start_time
    print(f"실행 시간:{running_time:2f}")

    
