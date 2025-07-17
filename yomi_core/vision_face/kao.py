import os
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, QObject
from PyQt5.QtGui import QPixmap


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

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def set_emotion_pixmap(self, pixmap: QPixmap):
        self.label.setPixmap(pixmap)


class FaceController(QObject):
    def __init__(self, gui: YomiFace):
        super().__init__()
        self.gui = gui  # GUI 객체를 받아서 제어

        self.emotions = ["joy", "sadness", "angry", "fear", "surprise", "disgust", "trust", "anticipation"]
        self.emotion_index = 0
        self.blinking = False
        self.img_index = 0

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_paths = self._load_image_paths()
        self.scaled_cache = {}

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_emotion)
        self.timer.start()

    def _load_image_paths(self):
        base = os.path.join(self.script_dir, 'blackface')
        return {
            emotion: (
                os.path.join(base, f'black_{emotion}D.png'),
                os.path.join(base, f'black_{emotion}S.png')
            ) for emotion in self.emotions
        }

    def update_emotion(self):
        if self.blinking:
            self.img_index = (self.img_index + 1) % 2
        else:
            self.img_index = 0

        emotion = self.emotions[self.emotion_index]
        cache_key = (emotion, self.img_index)

        if cache_key in self.scaled_cache:
            pixmap = self.scaled_cache[cache_key]
        else:
            path = self.image_paths[emotion][self.img_index]
            pixmap = QPixmap(path)
            if pixmap.isNull():
                print(f"[ERROR] Failed to load image: {path}")
                return
            pixmap = pixmap.scaled(
                self.gui.width(), self.gui.height(),
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            self.scaled_cache[cache_key] = pixmap

        self.gui.set_emotion_pixmap(pixmap)

    def set_blinking(self, value: bool):
        self.blinking = value

    def next_emotion(self):
        self.emotion_index = (self.emotion_index + 1) % len(self.emotions)
        print(f"[Controller] Emotion changed to {self.emotions[self.emotion_index]}")

    def set_emotion(self, emotion: str):
        """감정 이름으로 직접 감정을 설정"""
        if emotion in self.emotions:
            self.emotion_index = self.emotions.index(emotion)
            print(f"[Controller] Emotion manually set to '{emotion}'")
        else:
            print(f"[WARN] Unknown emotion: '{emotion}'")
 
    def stop(self):
        self.timer.stop()
        print("[Controller] Stopped")


# 앱 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 메인 GUI
    face = YomiFace()
    face.show()

    # 컨트롤러 스레드 + 객체
    controller_thread = QThread()
    controller = FaceController(face)
    controller.moveToThread(controller_thread)
    controller_thread.start()

    # 키보드 이벤트 처리 (face에서 직접)
    def key_handler(event):
        if event.key() == Qt.Key_Space:
            controller.next_emotion()
        elif event.key() == Qt.Key_B:
            controller.set_blinking(True)
        elif event.key() == Qt.Key_Y:
            controller.set_blinking(False)
        elif event.key() == Qt.Key_Q:
            controller.stop()
            controller_thread.quit()
            controller_thread.wait()
            face.close()

    face.keyPressEvent = key_handler

    sys.exit(app.exec_())
