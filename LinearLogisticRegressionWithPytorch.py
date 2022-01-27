# 使用Pytorch实现线性回归分析
# 当前的程序由于没使用归一化处理，所以需要使用非常小的学习率，从而导致学习效率非常低。使用LinearLogisticRegression.csv里的数据时，
# 由于两类数据分得很开，使用1000个epoch就可以很好的分类，所以计算速度慢的问题并不明显。
# 但是，当使用class_05_80.csv里的数据时，只有当epoch达到1000000时，才能得到较好的分类效果，此时完成一次计算需要经过漫长的等待！！！

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch import nn
from ID_DEFINE import *
from torch import optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    x0 = Data['x0'].values
    x1 = Data['x1'].values
    labels = Data['y'].values
    xValues = np.stack((x0, x1), axis=1)
    labels = labels.reshape(-1, 1)
    return xValues, labels


def MakeDataFile(filename):
    mu1 = -3 * torch.ones(2)
    mu2 = 3 * torch.ones(2)
    sigma1 = torch.eye(2) * 0.5
    sigma2 = torch.eye(2) * 2

    m1 = MultivariateNormal(mu1, sigma1)
    m2 = MultivariateNormal(mu2, sigma2)
    x1 = m1.sample((100,))
    x2 = m2.sample((100,))

    y = torch.zeros((200, 1))
    y[100:] = 1

    x = torch.cat([x1, x2], dim=0)
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]
    plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1])
    plt.scatter(x2.numpy()[:, 0], x2.numpy()[:, 1])
    plt.show()
    # print(x.shape, y.shape)
    dataArray = torch.cat([x, y], dim=1)
    dataArray = dataArray.detach().numpy()
    dataFrame = pd.DataFrame(dataArray, columns=['x0', 'x1', 'y'])
    dataFrame.to_csv(filename, mode='w', index=None, encoding='utf_8_sig')


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
    # filename = linearClassificationDataDir + "class_05_80.csv"
    filename = linearClassificationDataDir + "LinearLogisticRegression.csv"
    # MakeDataFile(filename)
    xVectors, labels = OpenDataFile(filename)
    xTensors = torch.tensor(xVectors, dtype=torch.float)
    labelTensors = torch.tensor(labels, dtype=torch.float)
    myLogisticRegression = LogisticRegression(2)
    optimizer = optim.SGD(myLogisticRegression.parameters(), lr=0.008)
    Loss = nn.BCELoss()
    for _ in range(maxEpoches):
        for i in range(int(len(xTensors) / batchSize)):
            input = xTensors[i * batchSize:(i + 1) * batchSize]
            labels = labelTensors[i * batchSize:(i + 1) * batchSize]
            optimizer.zero_grad()
            # myLogisticRegression.train()
            output = myLogisticRegression(input)
            loss = Loss(output, labels)
            loss.backward()
            optimizer.step()
    output = myLogisticRegression(xTensors)
    # print("output=", output)
    #
    # print("weight=",myLogisticRegression.linear.weight[0,0].item())

    w = -myLogisticRegression.linear.weight[0, 0] / myLogisticRegression.linear.weight[0, 1]
    bias = -myLogisticRegression.linear.bias[0] / myLogisticRegression.linear.weight[0, 1]
    # print("w,bias=", w, bias)
    for i, Tensor in enumerate(xTensors):
        if labelTensors[i, 0] == 1:
            plt.scatter(Tensor[0], Tensor[1], c='b', marker='*')
        else:
            plt.scatter(Tensor[0], Tensor[1], c='g', marker='.')

    x = torch.linspace(xTensors.min(), xTensors.max(), 100)
    y = x * w + bias
    plt.plot(x.detach().numpy(), y.detach().numpy(), 'r-')
    plt.show()
