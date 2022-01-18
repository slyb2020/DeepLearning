import pandas as pd
from ID_DEFINE import *
import matplotlib.pyplot as plt
import numpy as np


def OpenDataFile(fileName):
    Data = pd.read_csv(fileName)
    xValue = Data['x'].values
    yValue = Data['y'].values
    return xValue, yValue


if __name__ == '__main__':
    learningRate = 0.0001
    filename = linearRegressionDataDir + 'data.csv'
    xValue, yValue = OpenDataFile(filename)
    plt.plot(xValue, yValue, 'b.')
    a = 0
    b = 0
    delta = 1000
    maxDelta = 0.001
    iteration = 0
    while delta > maxDelta:
        for i, x in enumerate(xValue):
            y = a * x + b
            delta = yValue[i] - y
            gradienta = delta * x
            gradientb = delta
            a += learningRate * gradienta
            b += learningRate * gradientb
            print("y=", y)
            print("delta=", delta)
            if delta <= maxDelta:
                break
            iteration += 1
            print("iteration=", iteration)

    X = np.linspace(0, 10, 100)
    Y = a * X + b
    plt.plot(X, Y, 'r-')
    plt.show()
