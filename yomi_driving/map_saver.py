# yomi_driving/map_saver.py

import subprocess
import os

def save_map(map_name='map'):
    # SLAM으로 생성된 맵을 파일로 저장하는 함수 (.pgm, .yaml)
    save_path = os.path.expanduser(f'~/{map_name}')
    subprocess.run(['rosrun', 'map_server', 'map_saver', '-f', save_path])
    print(f'[✓] 맵 저장 완료: {save_path}.pgm / .yaml')