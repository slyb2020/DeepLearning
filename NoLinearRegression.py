# 基于维度增加的单一神经元非线性回归分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BatchGenerator import BatchGenerator
from ID_DEFINE import *


def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    xValues = Data['x0'].values
    labelValues = Data['y2'].values
    return xValues, labelValues


def RaiseDimension(x, dimension):
    dataArray = x.reshape(-1, 1)
    for i in range(dimension):
        dataArray1D = np.power(x, i + 2)
        dataArray = np.hstack((dataArray, dataArray1D.reshape(-1, 1)))
    return dataArray


def Loss(y, labels):
    loss = labels - y
    loss = loss * loss
    return np.mean(loss)


class NoLinearRegression:
    def __init__(self, dimension=2, learningRate=1, threshold=1e-10, maxEpochs=100000, miniBatchSize=5, momentum=0.9,
                 regularization=False, LOSS=Loss):
        self.dimension = dimension
        self.learningRate = learningRate
        self.threshold = threshold
        self.maxEpochs = maxEpochs
        self.regularization = regularization
        self.miniBatchSize = miniBatchSize
        self.momentum = momentum
        self.Loss = LOSS
        self.omega = np.array([0.0] * self.dimension)
        self.bias = 0.0
        self.minX = None
        self.rangeX = None

    def Regularization(self, inputX, labels):
        self.minX = inputX.min(0)
        maxX = inputX.max(0)
        self.rangeX = maxX - self.minX
        inputX = (inputX - self.minX) / self.rangeX
        return inputX, labels

    def UnRegularization(self):
        self.bias = self.bias - np.dot(self.omega * self.minX / self.rangeX, np.array([1] * self.dimension))
        self.omega = self.omega / self.rangeX

    def Predict(self, x):
        y = np.dot(x, self.omega) + self.bias
        return y

    def Fit(self, inputX, labels):
        gradientOmega = 0
        gradientBias = 0
        if self.regularization:
            inputX, labels = self.Regularization(inputX, labels)
        batchGenerator = BatchGenerator([inputX, labels], self.miniBatchSize, shuffle=True)
        for [xBatch, labelBatch], epoch in batchGenerator:
            y = self.Predict(xBatch)
            error = (labelBatch - y).reshape(-1, 1)
            loss = self.Loss(y, labelBatch)
            if (loss <= self.threshold) or (epoch >= self.maxEpochs):
                break
            gradientOmega = self.momentum * gradientOmega + np.mean(xBatch * error, axis=0)
            gradientBias = self.momentum * gradientBias + np.mean(error)
            self.omega += self.learningRate * gradientOmega
            self.bias += self.learningRate * gradientBias
        if self.regularization:
            self.UnRegularization()
        return loss, epoch


if __name__ == '__main__':
    filename = nolinearRegressionDataDir + 'ax2+bx+c.csv'
    xValues, labels = OpenDataFile(filename)
    print(xValues, labels)
    xVectors = RaiseDimension(xValues, 1)
    print("xVectors=", xVectors)
    myNoLinearRegression = NoLinearRegression(regularization=True)
    loss, epoch = myNoLinearRegression.Fit(xVectors, labels)
    print("loss,epoch=", loss, epoch)
    print("omega=", myNoLinearRegression.omega)
    print("bias=", myNoLinearRegression.bias)
    plt.scatter(xValues,labels,c='b',marker='.')
    minX = xValues.min()
    maxX = xValues.max()
    xValues = np.linspace(minX-1,maxX+1,100)
    xPredict = RaiseDimension(xValues,1)
    yPredict = myNoLinearRegression.Predict(xPredict)
    plt.plot(xValues,yPredict,'r-')
    plt.show()
