#当使用未归一化数据时，由于输入数据没有归一化处理，所以学习率只能取得非常小，在取0.000016时，需要迭代1564个epoch才能得到较好的解。
#当使用归一化数据时，由于输入数据有归一化处理，所以学习率可以较大，在取1时，甚至在第0个epoch中即可求得非常好的解。
#可见归一化对于缩短训练时间的效果非常显著
#20220119 11:44在ZenBook上增加归一化处理功能
import numpy as np
import pandas as pd

from ID_DEFINE import *


def OpenDataFile(fileName):
    Data = pd.read_csv(fileName)
    x0 = Data['x0'].values
    x1 = Data['x1'].values
    xValues = np.stack((x0, x1), axis=1)
    labels = Data['y2'].values
    return xValues, labels


class LinearRegression2D:
    def __init__(self, dimension=2, learningRate=1, maxLoss=1e-10, epoch=100000):
        self.dimension = dimension
        self.learningRate = learningRate
        self.maxLoss = maxLoss
        self.epoch = epoch
        self.omega = np.array([0.0] * self.dimension)
        self.bias = 0.0

    def Predict(self, x):
        y = np.dot(x, self.omega) + self.bias
        return y

    def Loss(self, y, labels):
        loss = labels - y
        loss = loss * loss
        return loss

    def Fit(self, inputX, labels):
        for i in range(self.epoch):
            for j, x in enumerate(inputX):
                y = self.Predict(x)
                error = labels[j] - y
                loss = self.Loss(y, labels[j])
                if loss <= self.maxLoss:
                    return loss, i
                self.omega += self.learningRate * x * error
                self.bias += self.learningRate * error
        return loss, self.epoch


if __name__ == '__main__':
    # filename = linearRegressionDataDir + 'LinearRegression2D.csv'
    filename = linearRegressionDataDir + 'data_Regression_regular.csv'
    xVectors, labels = OpenDataFile(filename)
    myLinearRegression2D = LinearRegression2D()
    loss, epoch = myLinearRegression2D.Fit(xVectors, labels)
    print("loss,epoch=", loss, epoch)
    print("omega=", myLinearRegression2D.omega)
    print("bias=", myLinearRegression2D.bias)
