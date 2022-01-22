import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BatchGenerator import BatchGenerator
from ID_DEFINE import *


def OpenDataFile(fileName):
    Data = pd.read_csv(fileName)
    x0 = Data['x0'].values
    x1 = Data['x1'].values
    xValues = np.stack((x0, x1), axis=1)
    labels = Data['y'].values
    return xValues, labels


def Loss(y, labels):
    loss = labels - y
    loss = loss * loss
    return np.mean(loss)


def Activator(x):
    return 1 if x > 0 else 0


class Perceptron:
    def __init__(self, dimension=2, learningRate=1, threshold=1e-10, maxEpochs=10000, miniBatchSize=5, momentum=0.9,
                 regularization=False, LOSS=Loss, Activator=Activator):
        self.dimension = dimension
        self.learningRate = learningRate
        self.threshold = threshold
        self.maxEpochs = maxEpochs
        self.regularization = regularization
        self.miniBatchSize = miniBatchSize
        self.momentum = momentum
        self.Loss = LOSS
        self.Activator = np.vectorize(Activator)  # numpy的vectorize函数可以将目标函数适量话。这个函数非常有用一定要记住、用熟
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
        return self.Activator(y)

    def Boundary(self, inputX):
        self.a = -self.omega[0] / self.omega[1]
        self.b = -self.bias / self.omega[1]
        outputY = self.a * inputX + self.b
        return outputY

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
            if loss <= self.threshold:
                break
            if epoch >= self.maxEpochs:
                break
            gradientOmega = self.momentum * gradientOmega + np.mean(xBatch * error, axis=0)
            gradientBias = self.momentum * gradientBias + np.mean(error)
            self.omega += self.learningRate * gradientOmega
            self.bias += self.learningRate * gradientBias
        if self.regularization:
            self.UnRegularization()
        return loss, epoch


if __name__ == '__main__':
    # filename = linearClassificationDataDir + 'class_05_80.csv'
    filename = linearClassificationDataDir + 'admissionData.csv'
    xVectors, labels = OpenDataFile(filename)
    myPerceptron = Perceptron(regularization=True, miniBatchSize=100)
    loss, epoch = myPerceptron.Fit(xVectors, labels)
    print("loss,epoch=", loss, epoch)
    print("omega=", myPerceptron.omega)
    print("bias=", myPerceptron.bias)
    inputX = np.linspace(0, 100, 500)
    outputYY = myPerceptron.Boundary(inputX)
    plt.scatter(xVectors[:, 0], xVectors[:, 1])
    for i, vector in enumerate(xVectors):
        if labels[i] == 0:
            plt.plot(vector[0], vector[1], 'r.')
        else:
            plt.plot(vector[0], vector[1], 'b*')
    plt.plot(inputX, outputYY, 'r-')
    plt.show()
    print("a,b=", myPerceptron.a, myPerceptron.b)
