# 使用Pytorch实现线性回归分析
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch import nn
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
    maxEpoches = 10000
    batchSize = 100
    filename = linearClassificationDataDir + "class_05_80.csv"
    xVectors, labels = OpenDataFile(filename)
    xTensors = torch.tensor(xVectors, dtype=torch.float)
    labelTensors = torch.tensor(labels, dtype=torch.float)
    myLogisticRegression = LogisticRegression(2)
    optimizer = optim.SGD(myLogisticRegression.parameters(), lr=0.001)
    Loss = nn.BCELoss()
    for _ in range(maxEpoches):
        for i in range(int(len(xTensors) / batchSize)):
            input = xTensors[i * batchSize:(i + 1) * batchSize]
            labels = labelTensors[i * batchSize:(i + 1) * batchSize]
            optimizer.zero_grad()
            myLogisticRegression.train()
            output = myLogisticRegression(input)
            loss = Loss(output, labels)
            loss.backward()
            optimizer.step()
    output = myLogisticRegression(xTensors)
    print("output=", output)

    print("weight=",myLogisticRegression.linear.weight[0,0].item())

    w = -myLogisticRegression.linear.weight[0,0] / myLogisticRegression.linear.weight[0,1]
    bias = -myLogisticRegression.linear.bias[0] / myLogisticRegression.linear.weight[0,1]
    print("w,bias=", w, bias)
    for i,Tensor in enumerate(xTensors):
        if labelTensors[i,0] == 1:
            plt.scatter(Tensor[0], Tensor[1], c='b', marker='*')
        else:
            plt.scatter(Tensor[0], Tensor[1], c='g', marker='.')

    x = torch.linspace(0,100,100)
    y = x * w + bias
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'r-')
    plt.show()
