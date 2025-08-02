# yomi_driving/map_editor.py

import subprocess
    
def open_kolourpaint():
    # 저장된 맵(.pgm) 파일을 편집하기 위해 kolourpaint 이미지 편집기 실행
    subprocess.Popen(['kolourpaint'])  # GUI로 실행됨
