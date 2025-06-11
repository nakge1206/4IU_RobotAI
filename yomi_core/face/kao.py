import os
import threading
import tkinter as tk
from PIL import Image, ImageTk

class EmotionApp:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_paths = self._load_image_paths()
        self.mind = 'joy'
        self.img_index = 0
        self.tk_img = None

        self.window = tk.Tk()
        self.window.attributes('-fullscreen', True)

        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.label = tk.Label(self.window)
        self.label.pack()

        self._bind_keys()
        self._start_input_thread()

        self.show_emotion()

        self.window.mainloop()

    def _load_image_paths(self):
        base = os.path.join(self.script_dir, 'face', 'blackface')
        emotions = ['joy', 'angry', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust']
        return {
            emotion: (
                os.path.join(base, f'black_{emotion}D.png'),
                os.path.join(base, f'black_{emotion}S.png')
            ) for emotion in emotions
        }

    def _bind_keys(self):
        self.window.bind("<space>", self.toggle_emotion)
        self.window.bind("q", self.quit_app)
        self.window.bind("<Key>", self.change_emotion_key)

    def _start_input_thread(self):
        thread = threading.Thread(target=self._read_input_loop, daemon=True)
        thread.start()

    def _read_input_loop(self):
        while True:
            new_mind = input().strip().lower()
            if new_mind in self.image_paths:
                self.mind = new_mind
                self.img_index = 0
                self.show_emotion()

    def load_image(self, path):
        img = Image.open(path)
        img = img.resize((self.screen_width, self.screen_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def show_emotion(self, event=None):
        self.img_index = (self.img_index + 1) % 2
        current_paths = self.image_paths.get(self.mind)
        if current_paths:
            path = current_paths[self.img_index]
            self.tk_img = self.load_image(path)
            self.label.config(image=self.tk_img)

    def toggle_emotion(self, event=None):
        self.show_emotion()

    def change_emotion_key(self, event=None):
        key = event.char.strip().lower()
        if key in self.image_paths:
            self.mind = key
            self.img_index = 0
            self.show_emotion()

    def quit_app(self, event=None):
        self.window.quit()


if __name__ == '__main__':
    EmotionApp()
