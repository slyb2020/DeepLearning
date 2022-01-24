# 基于Pytorch的非线性回归分析
import pandas as pd
from torchvision import datasets, transforms
import torch
from torch import nn
import numpy as np
from torchsummary import summary
from BatchGenerator import BatchGenerator
from ID_DEFINE import *


def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    xValues = Data['x0'].values
    labelValues = Data['y2'].values
    return xValues, labelValues


class NoLinearRegressionNet(nn.Module):
    def __init__(self, inputDimension, hide1Dimension, outputDimension):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(inputDimension, hide1Dimension), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hide1Dimension, outputDimension))

    def forward(self, x):
        x = self.layer1(x)
        output = self.layer2(x)
        return output


if __name__ == "__main__":
    filename = nolinearRegressionDataDir + 'ax2+bx+c.csv'
    xValues, labels = OpenDataFile(filename)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batchSize = 5
    model = NoLinearRegressionNet(1, 8, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    learningRate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    numEpoches = 10
    trainLoader = BatchGenerator([xValues, labels], batchSize, shuffle=True)
    exEpoch = 0
    train_accuracy_total = 0
    train_correct = 0
    train_loss_sum = 0
    threshold = 1e-4
    for [xBatch, labelBatch], epoch in trainLoader:
        if epoch >= numEpoches:
            break
        if exEpoch != epoch:
            exEpoch = epoch
            if(train_loss_sum<=threshold):
                break
            train_loss_sum = 0
            print("epoch=",epoch)
        xBatch = torch.tensor(xBatch.reshape(-1, 1), dtype=torch.float)
        # print(xBatch)
        labelBatch = torch.tensor(labelBatch.reshape(-1, 1), dtype=torch.float)
        # print(labelBatch)
        model.train()
        xBatch = xBatch.to(device)  # 这句话现在好使了，因为xBatch已经是tensor型变量了
        labelBatch = labelBatch.to(device)
        outputs = model(xBatch)
        loss = criterion(outputs, labelBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        # _, predicts = torch.max(outputs.data, 1)
        # train_accuracy_total += labelBatch.size(0)
        # train_correct += (predicts == labelBatch).cpu().sum().item()
    print("finish training")
    print('epoch %d,loss %.5f'%(epoch,train_loss_sum/batchSize))
