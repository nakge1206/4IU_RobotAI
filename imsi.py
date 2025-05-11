# 파이썬에서 실행해보세요
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(torch.cuda.is_available())  # True면 정상
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4070
