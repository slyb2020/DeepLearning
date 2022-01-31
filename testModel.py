# 测试模型存储与读取方法
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader as Loader
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # python的进度条模块

from ID_DEFINE import *

# modelFileName = modelDir + 'FashionMNIST.model'
# myModel = torch.load(modelFileName)
# for parameter in myModel.conv1.parameters():
#     print(parameter.cpu().detach().numpy())
# # 以上代码运行出错，说明torch.save（）并没有存储网络的全部参数，只是存储了一部分，具体还需要进一步研究。。。。

# 要想让读取的模型能用需要先定义模型结构，然后再读取模型
class FashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(self.fc3(x),dim=1)
        return x


myModel = FashionMNIST()
modelFileName = modelDir + 'FashionMNIST.model'
myModel = torch.load(modelFileName)
print(modelFileName)
print(myModel)
for parameter in myModel.conv1.parameters():
    print(parameter.cpu().detach().numpy())

