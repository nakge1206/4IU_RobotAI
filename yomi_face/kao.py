import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
import queue

class Yomi_face:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_paths = self._load_image_paths()
        self.emotion = 'joy'
        self.img_index = 0
        self.tk_img = None

        self.window = tk.Tk()
        self.window.attributes('-fullscreen', True)

        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.label = tk.Label(self.window)
        self.label.pack()

        self._bind_keys()

        self.blinking = False

    def start(self):
        """시작"""
        self.show_emotion()
        self.window.mainloop()
    
    def stop(self):
        """중지"""
        self.window.quit()

    def set_blinking(self, value: bool):
        self.blinking = value

    def set_emotion(self, emotion):
        """input : emotion(angry, anticipation, disgust, fear, joy, sadness, surprise, trust)"""
        if emotion in self.image_paths:
            self.emotion = emotion
            self.img_index = 0

    # def change_face(self, emotion):
    #     """input : emotion(angry, anticipation, disgust, fear, joy, sadness, surprise, trust)"""
    #     if emotion in self.image_paths:
    #         self.emotion = emotion
    #         self.img_index = 0
    #         self.show_emotion()
    
    def toggle_emotion(self, on_tts=False):
        self.show_emotion(on_tts)

    def show_emotion(self):
        """현재 감정에 따라 이미지를 번갈아 표시"""
        if self.blinking:
            self.img_index = (self.img_index + 1) % 2
            current_paths = self.image_paths.get(self.emotion)
            if current_paths:
                path = current_paths[self.img_index]
                self.tk_img = self.load_image(path)
                self.label.config(image=self.tk_img)
        else:
            self.img_index = 0
            current_paths = self.image_paths.get(self.emotion)
            if current_paths:
                path = current_paths[self.img_index]
                self.tk_img = self.load_image(path)
                self.label.config(image=self.tk_img)
        
        # 0.2초 후에 다시 자신을 호출 (반복 깜빡임)
        self.window.after(100, self.show_emotion)

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

    def quit_app(self, event=None):
        self.window.quit()


if __name__ == '__main__':
    Yomi_face().start()
