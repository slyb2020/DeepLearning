# 基于keras和tensorflow组成的神经网络的非线性回归分析
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from ID_DEFINE import *


def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    xValues = Data['x0'].values
    labelValues = Data['y2'].values
    return xValues, labelValues


if __name__ == '__main__':
    filename = nolinearRegressionDataDir + 'ax2+bx+c.csv'
    xValues, labels = OpenDataFile(filename)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        # tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MSE,
                  metrics=['accuracy'])

    model.fit(xValues, labels, epochs=50)

    minX = xValues.min()
    maxX = xValues.max()
    xPredict = np.linspace(minX - 1, maxX + 1, 100)
    yPredict = model.predict(xPredict)

    plt.scatter(xValues, labels, c='b', marker='.')
    plt.plot(xPredict, yPredict, 'r-')
    plt.show()
