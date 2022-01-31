# 使用Pytorch基于CNN编写FasionMNIST识别程序
# 包含基于TensorBoard的数据可视化代码

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


def accuracy(predict, label):
    predictLabel = torch.argmax(predict, 1)
    correct = sum(predictLabel == label).to(torch.float)
    # acc = correct / float(len(pred))
    return correct, len(predict)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batchSize = 64
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainDataset = torchvision.datasets.FashionMNIST('D:/WorkSpace/DataSet', download=True, train=True,
                                                     transform=transformer)
    testDataset = torchvision.datasets.FashionMNIST('D:/WorkSpace/DataSet', download=True, train=False,
                                                    transform=transformer)
    trainLoader = Loader.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = Loader.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    # labelClasses = ("衬衫", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "靴子")
    labelClasses = (
        "T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")
    modelFileName = modelDir + 'FashionMNIST.model'
    isExists = os.path.exists(modelFileName)
    if isExists:
        myModel = torch.load(modelFileName)
        if torch.cuda.is_available():
            myModel = myModel.cuda()
        myModel.eval()
        correctNumberVerify, totalNumberVerify = 0.0, 0.0
        with torch.no_grad():
            for data, label, in testLoader:
                data = data.to(device)  # 这句话现在好使了，因为xBatch已经是tensor型变量了
                label = label.to(device)
                output = myModel(data)
                correctNumber, totalNumber = accuracy(output, label)
                correctNumberVerify += correctNumber
                totalNumberVerify += totalNumber
        print("Accuracy is ", (correctNumberVerify / totalNumberVerify).cpu().detach().numpy())
    else:
        print("Start training")
        myModel = FashionMNIST()
        if torch.cuda.is_available():
            myModel = myModel.cuda()
        # print(myModel)
        Loss = nn.CrossEntropyLoss()
        Optimizer = optim.SGD(myModel.parameters(), lr=1e-3, momentum=0.9)
        # threshold = 0.0
        # runningLoss = 0.0
        for epoch in tqdm(range(5)):
            myModel.eval()
            correctNumberVerify, totalNumberVerify, lossVerify = 0.0, 0.0, 0.0
            with torch.no_grad():
                for data, target in testLoader:
                    data = data.to(device)  # 这句话现在好使了，因为xBatch已经是tensor型变量了
                    target = target.to(device)
                    output = myModel(data)
                    loss = Loss(output, target)
                    lossVerify += loss
                    correctnumber, totalNumber = accuracy(output, target)
                    correctNumberVerify += correctnumber
                    totalNumberVerify += totalNumber
            print("Accuracy is ", (correctNumberVerify / totalNumberVerify).cpu().detach().numpy())
            myModel.train()
            correctNumberTrain, totalNumberTrain, lossTrain = 0., 0., 0.,
            for data, target in trainLoader:
                data = data.to(device)  # 这句话现在好使了，因为xBatch已经是tensor型变量了
                target = target.to(device)
                Optimizer.zero_grad()  # 梯度初始化为0
                output = myModel(data)
                loss = Loss(output, target)
                lossTrain += loss
                loss.backward()
                Optimizer.step()
                correctNumber, totalNumber = accuracy(output, target)
                correctNumberTrain += correctNumber
                totalNumberTrain += totalNumber
            # print("Accuracy is ", (correctNumberVerify / totalNumberVerify).detach().numpy())
        print("Accuracy is ", (correctNumberVerify / totalNumberVerify).cpu().detach().numpy())
        print("Finish training")
        torch.save(myModel, modelDir + 'FashionMNIST.model')
    for parameter in myModel.conv1.parameters():
        print(parameter.cpu().detach().numpy())
