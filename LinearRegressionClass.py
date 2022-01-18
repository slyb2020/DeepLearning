import pandas as pd
from ID_DEFINE import *
import matplotlib.pyplot as plt
import numpy as np

def OpenDataFile(fileName):
    Data = pd.read_csv(fileName)
    xValue = Data['x'].values
    yValue = Data['y'].values
    return xValue, yValue


class MyLinearRegression():
    def __init__(self,dimension=1, learningRate=0.0001, maxLoss=1e-6, epoch=10000):
        self.dimension = dimension
        self.learningRate = learningRate
        self.maxLoss = maxLoss
        self.epoch = epoch
        self.omega = np.array([0.0]*dimension)
        self.bias = 0.0

    def predict(self, x):
        if xVectors.shape[1] != self.dimension:
            raise ValueError("输入数据的维数不对！")
        y = np.dot(x,self.omega)+self.bias
        return y

    def Loss(self, y, label):
        loss = (label - y) * (label - y)
        return loss

    def fit(self, xVectors ,labels):
        for i in range(self.epoch):
            for j, x in enumerate(xVectors):
                y = self.predict(x)
                error = labels[j] - y
                loss = self.Loss(y,labels[j])
                if loss <= self.maxLoss:
                    return loss, i
                else:
                    self.omega += self.learningRate * x * error
                    self.bias += self.learningRate * error
        return loss, self.epoch


if __name__ == '__main__':
    filename = linearRegressionDataDir + 'data.csv'
    xValue, yValue = OpenDataFile(filename)
    plt.plot(xValue, yValue, 'b.')
    xVectors = xValue.reshape(-1,1)
    myLinearRegression = MyLinearRegression(1)
    loss,epoch = myLinearRegression.fit(xVectors,yValue)
    print("loss,epoch=",loss,epoch)
    X = np.linspace(0, 10, 100).reshape(-1,1)
    Y = myLinearRegression.predict(X)
    plt.plot(X, Y, 'r-')
    plt.show()
