# 当使用未归一化数据时，由于输入数据没有归一化处理，所以学习率只能取得非常小，在取0.000016时，需要迭代1564个epoch才能得到较好的解。
# 当使用归一化数据时，由于输入数据有归一化处理，所以学习率可以较大，在取1时，甚至在第0个epoch中即可求得非常好的解。
# 可见归一化对于缩短训练时间的效果非常显著
# 20220119 11:44在ZenBook上增加归一化处理功能
# 针对data_Regression_unregular.csv中的数据，如果不使用归一化处理，learningRate需要设置为0.00001，而且还需要经过2915个epoch的迭代
# 才能得到一个比较准确的结果；如果使用归一化处理，则learningRate可以简单的设置为1，只需要1个epoch就能得到较为理想的结果.
# 可见归一化处理对于提高线性回归收敛速度具有非常显著的效果！
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
    def __init__(self, dimension=2, learningRate=1, threshold=1e-10, max_epochs=100000, regularization=False):
        self.dimension = dimension
        self.learningRate = learningRate
        self.threshold = threshold
        self.max_epochs = max_epochs
        self.regularization = regularization
        self.omega = np.array([0.0] * self.dimension)
        self.bias = 0.0
        self.minX = None
        self.rangeX = None

    def Regularization(self, inputX, labels):
        self.minX = inputX.min(0)
        maxX = inputX.max(0)
        self.rangeX = maxX - self.minX
        print("minX=", self.minX)
        inputX = (inputX - self.minX) / self.rangeX
        return inputX, labels

    def UnRegularization(self):
        self.bias = self.bias - np.dot(self.omega * self.minX / self.rangeX, np.array([1] * self.dimension))
        self.omega = self.omega / self.rangeX

    def Predict(self, x):
        y = np.dot(x, self.omega) + self.bias
        return y

    def Loss(self, y, labels):
        loss = labels - y
        loss = loss * loss
        return loss

    def Fit(self, inputX, labels):
        if self.regularization:
            inputX, labels = self.Regularization(inputX, labels)
        for i in range(self.max_epochs):
            for j, x in enumerate(inputX):
                y = self.Predict(x)
                error = labels[j] - y
                loss = self.Loss(y, labels[j])
                if loss <= self.threshold:
                    if self.regularization:
                        self.UnRegularization()
                    return loss, i
                self.omega += self.learningRate * x * error
                self.bias += self.learningRate * error
        if self.regularization:
            self.UnRegularization()
        return loss, self.max_epochs


if __name__ == '__main__':
    filename = linearRegressionDataDir + 'data_Regression_unregular.csv'
    # filename = linearRegressionDataDir + 'data_Regression_regular.csv'
    xVectors, labels = OpenDataFile(filename)
    myLinearRegression2D = LinearRegression2D(regularization=True)
    loss, epoch = myLinearRegression2D.Fit(xVectors, labels)
    print("loss,epoch=", loss, epoch)
    print("omega=", myLinearRegression2D.omega)
    print("bias=", myLinearRegression2D.bias)
