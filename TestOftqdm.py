# python 进度条模块（tqdm）测试
from tqdm import tqdm
import time


for i in tqdm(range(100)):
    time.sleep(0.2)