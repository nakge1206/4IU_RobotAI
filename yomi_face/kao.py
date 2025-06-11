import os
import threading
import tkinter as tk
from PIL import Image, ImageTk

class Yomi_face:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_paths = self._load_image_paths()
        self.emotion = 'joy'
        self.img_index = 0
        self.tk_img = None

        self.is_tts = False

        self.window = tk.Tk()
        self.window.attributes('-fullscreen', True)

        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.label = tk.Label(self.window)
        self.label.pack()

        self._bind_keys()
        self._start_input_thread()

    def start(self):
        """시작"""
        self.show_emotion()
        self.window.mainloop()
    
    def stop(self):
        """중지"""
        self.window.quit()

    def change_face(self, emotion):
        """input : emotion(angry, anticipation, disgust, fear, joy, sadness, surprise, trust)"""
        if emotion in self.image_paths:
            self.emotion = emotion
            self.img_index = 0
            self.show_emotion()
    
    def toggle_emotion(self, on_tts=False):
        self.window.after(0, lambda: self._safe_toggle_emotion(on_tts))
        # self.show_emotion(on_tts)
    
    def _safe_toggle_emotion(self, on_tts=False):
        self.show_emotion(on_tts)

    def show_emotion(self, on_tts=None):
        """현재 감정에 따라 이미지를 번갈아 표시"""
        self.img_index = (self.img_index + 1) % 2
        current_paths = self.image_paths.get(self.emotion)
        if current_paths:
            path = current_paths[self.img_index]
            self.tk_img = self.load_image(path)
            self.label.config(image=self.tk_img)
        
        # 1초 후에 다시 자신을 호출 (반복 깜빡임)
        if on_tts:
            self.window.after(400, lambda: self.show_emotion(on_tts=True))

    def load_image(self, path):
        """이미지 불러온 후, Tk에 사용할 이미지로 변경"""
        img = Image.open(path)
        img = img.resize((self.screen_width, self.screen_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def change_emotion_key(self, event=None):
        key = event.char.strip().lower()
        if key in self.image_paths:
            self.mind = key
            self.img_index = 0
            self.show_emotion()

    def _load_image_paths(self):
        """경로 내 emotion.png 경로를 return하는 내부 함수"""
        base = os.path.join(self.script_dir, 'blackface')
        emotions = ['joy', 'angry', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust']
        return {
            emotion: (
                os.path.join(base, f'black_{emotion}D.png'),
                os.path.join(base, f'black_{emotion}S.png')
            ) for emotion in emotions
        }

    def _bind_keys(self):
        """space : emotion_toggle, q : quit, <key> : change_emotion"""
        self.window.bind("<space>", self.toggle_emotion)
        self.window.bind("q", self.quit_app)
        self.window.bind("<Key>", self.change_emotion_key)

    def _start_input_thread(self):
        """별도의 스레드로 입력 감지"""
        thread = threading.Thread(target=self._read_input_loop, daemon=True)
        thread.start()

    def _read_input_loop(self):
        """입력된 emotion으로 사진 변경"""
        while True:
            new_mind = input().strip().lower()
            if new_mind in self.image_paths:
                self.mind = new_mind
                self.img_index = 0
                self.show_emotion()

    def quit_app(self, event=None):
        self.window.quit()


if __name__ == '__main__':
    Yomi_face().start()
