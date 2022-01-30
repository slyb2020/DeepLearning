# 使用Pytorch基于CNN编写FasionMNIST识别程序

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

from ID_DEFINE import *


class FashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,15,5)
        self.fc1 = nn.Linear(14*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(self.fc3(x),dim=1)
        return x


if __name__ == "__main__":
    batchSize = 4
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainDataset = torchvision.datasets.FashionMNIST('D:/WorkSpace/DataSet', download=True, train=True, transform=transformer)
    testDataset = torchvision.datasets.FashionMNIST('D:/WorkSpace/DataSet', download=True, train=False, transform=transformer)
    trainLoader = Loader.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = Loader.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    labelClasses = ("衬衫", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "靴子")
    myFashionMNIST = FashionMNIST()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(myFashionMNIST.parameters(), lr=1, momentum=0.9)

    writer = SummaryWriter(tensorboardDir)
    dataIter = iter(trainLoader)
    images, labels = next(dataIter)

    imgGrid = torchvision.utils.make_grid(images)

    writer.add_image('four_fashion_mnist_images', imgGrid)
    print("Done")
