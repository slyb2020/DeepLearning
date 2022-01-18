import pandas as pd
from ID_DEFINE import *
import matplotlib.pyplot as plt
import numpy as np

def OpenDataFile(fileName):
    Data = pd.read_csv(fileName)
    x0 = Data['x0'].values
    x1 = Data['x1'].values
    xValues = np.stack((x0,x1),axis=1)
    labels = Data['y'].values
    return xValues,labels

class LinearRegression2D():
    def __init__(self,dimension=2,learningRate=0.01,maxLoss=1e-10,epoch=10000):
        self.dimension = dimension
        self.learningRate = learningRate
        self.maxLoss = maxLoss
        self.epoch = epoch
        self.omega = np.array([0.0]*self.dimension)
        self.bias = 0.0

    def Predict(self,x):
        y = np.dot(x,self.omega) + self.bias
        return y

    def Loss(self, y, labels):
        loss = (labels - y) * (labels - y)
        return loss

    def Fit(self, inputX, labels):
        for i in range(self.epoch):
            for j, x in enumerate(inputX):
                y = self.Predict(x)
                error = labels[j]- y
                loss = self.Loss(y,labels[j])
                if loss <= self.maxLoss:
                    return loss, i
                self.omega += self.learningRate * x * error
                self.bias += self.learningRate * error
        return loss, self.epoch

if __name__ == '__main__':
    filename = linearRegressionDataDir+'LinearRegression2D.csv'
    xVectors,labels = OpenDataFile(filename)
    myLinearRegression2D = LinearRegression2D()
    loss,epoch = myLinearRegression2D.Fit(xVectors,labels)
    print("loss,epoch=",loss,epoch)
    print("omega=",myLinearRegression2D.omega)
    print("bias=",myLinearRegression2D.bias)
