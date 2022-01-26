# 使用Pytorch实现线性回归分析
import torch
import numpy as np
import pandas as pd
from  torch import nn
from ID_DEFINE import *
from torch import optim

def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    x0 = Data['x0'].values
    x1 = Data['x1'].values
    labels = Data['y'].values
    xValues = np.stack((x0, x1), axis=1)
    labels = labels.reshape(-1, 1)
    return xValues, labels


class LogisticRegression(nn.Module):
    def __init__(self, inDimension):
        super().__init__()
        self.linear = nn.Linear(inDimension, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputVectors):
        output = self.linear(inputVectors)
        output = self.sigmoid(output)
        return output


if __name__ == "__main__":
    inDimension = 2
    outDimension = 1
    maxEpoches = 10
    batchSize = 20
    filename = linearClassificationDataDir + "class_05_80.csv"
    xVectors, labels = OpenDataFile(filename)
    xTensors = torch.tensor(xVectors, dtype=torch.float)
    labelTensors = torch.tensor(labels, dtype=torch.float)
    myLogisticRegression = LogisticRegression(2)
    optimizer = optim.SGD(myLogisticRegression.parameters(), lr=0.03)
    Loss = nn.BCELoss()
    for _ in range(maxEpoches):
        for i in range(int(len(xTensors)/batchSize)):
            input = xTensors[i*batchSize:(i+1)*batchSize]
            labels = labelTensors[i*batchSize:(i+1)*batchSize]
            optimizer.zero_grad()
            output = myLogisticRegression(input)
            loss = Loss(output, labels)
            loss.backward()
            optimizer.step()
    output = myLogisticRegression.forward(xTensors)
    print("output=",output)