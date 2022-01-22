import numpy as np
import pandas as pd
from ID_DEFINE import *
import matplotlib.pyplot as plt

from BatchGenerator import BatchGenerator


def Relu(inputVectors):
    outputVectors = np.array([0 if d < 0 else d for d in inputVectors]).reshape(-1, 1)
    return outputVectors


class MultyLayerNeuralNetwork:
    def __init__(self, inputShape, shape, activators, learningRate=0.01, threshold=1e-5, maxEpoch=300, regularization=False,
                 miniBatchSize=10, momentum=0.6, softmax=False, decay_power=0.2, verbose=False):  # 这里的shape
        # 是除了输入层以外的其它层的shape列表
        if len(activators) != len(shape):
            raise Exception("激活函数数量与神经网络层数不匹配，程序无法运行！请在下次运行程序前确保为每层网络都指定激活函数！")
        self.inputShape = inputShape
        self.shape = shape
        self.threshold = threshold
        self.maxEpoch = maxEpoch
        self.regularization = regularization
        self.miniBatchSize = miniBatchSize
        self.momentum = momentum
        self.isSoftmax = softmax
        self.learningRate = float(learningRate)
        self.effectiveLearningRate = self.learningRate
        self.decay_power = float(decay_power)

        self.depth = len(shape)
        self.outputs = [np.mat(0)] * (self.depth + 1)  # outputs[0]是输入层的输出。
        self.activityLevels = [np.mat(0)] * self.depth
        self.deltas = [np.mat(0)] * self.depth

        self.weights = [np.mat(0)] * self.depth
        self.biases = [np.mat(0)] * self.depth
        self.acc_weights_delta = [np.mat(0)] * self.depth
        self.acc_biases_delta = [np.mat(0)] * self.depth

        self.weights[0] = np.mat(np.random.random((shape[0], inputShape)) / 100.)
        self.biases[0] = np.mat(np.random.random((shape[0], 1)) / 100.)
        for i, idx in enumerate(self.shape[1:]):
            self.weights[i + 1] = np.mat(np.random.random((idx, self.shape[i])) / 100.)
            self.biases[i + 1] = np.mat(np.random.random((idx, 1)) / 100.)

        self.activatorFunc = []
        self.activatorFuncDiff = []
        for activator in activators:
            if activator.upper() == 'RELU':
                self.activatorFunc.append(np.vectorize(self.Relu))
                self.activatorFuncDiff.append(np.vectorize(self.ReluDiff))
            elif activator.upper() == 'IDENTITY':
                self.activatorFunc.append(np.vectorize(self.Identity))
                self.activatorFuncDiff.append(np.vectorize(self.IdentityDiff))

    def Compute(self, inputX):
        result = inputX
        for idx in range(self.depth):
            self.outputs[idx] = result  # outputs[0]是输入层的输出，而idx=0是计算出的result，在idx=1时赋值给outputs[idx]，正好！
            self.activityLevels[idx] = self.weights[idx] * result + self.biases[idx]
            result = self.activatorFunc[idx](self.activityLevels[idx])
        self.outputs[self.depth] = result
        return self.softmax(result) if self.isSoftmax else result

    def Predict(self, x):
        return self.Compute(np.mat(x).T).T.A  # 输出好结果为一列数，行数是输入的数据条数，列数是网络输出层神经元个数

    def BP(self, error):
        temp = error.T
        for idx in range(self.depth)[::-1]:
            delta = np.multiply(temp, self.activatorFuncDiff[idx](self.activityLevels[idx]).T)
            self.deltas[idx] = delta  # 这个deltas应该叫做寻优目标对神经网络各层激活水平的偏导数的行向量组
            temp = delta * self.weights[idx]  # 这是点乘求内积，得到额是一个列向量，它的行数与上一层网络的元素数相等。

    def Update(self):

        self.effectiveLearningRate = self.learningRate / np.power(self.iteration, self.decay_power)

        for idx in np.arange(0, self.depth):
            # current gradient
            weightsGradient = -self.deltas[idx].T * self.outputs[idx].T / self.deltas[idx].shape[0] + \
                              self.regularization * self.weights[idx]
            biasesGradient = -np.mean(self.deltas[idx].T, axis=1) + self.regularization * self.biases[idx]

            # accumulated delta
            self.acc_weights_delta[idx] = self.acc_weights_delta[
                                              idx] * self.momentum - self.effectiveLearningRate * weightsGradient
            self.acc_biases_delta[idx] = self.acc_biases_delta[
                                             idx] * self.momentum - self.effectiveLearningRate * biasesGradient

            self.weights[idx] = self.weights[idx] + self.acc_weights_delta[idx]
            self.biases[idx] = self.biases[idx] + self.acc_biases_delta[idx]

    def Fit(self, inputX, labels):
        batchGenerator = BatchGenerator([inputX, labels], self.miniBatchSize, shuffle=True)
        lossList = []
        self.iteration = 0
        exEpoch = 0
        for [xBatch, labelBatch], epoch in batchGenerator:
            if exEpoch != epoch:
                exEpoch = epoch
                loss = np.mean(lossList)
                lossList = []
                if (epoch >= self.maxEpoch) or (loss <= self.threshold):
                    break
            xBatch = xBatch.T
            labelBatch = labelBatch.T
            results = self.Compute(xBatch)
            errors = labelBatch - results
            if self.isSoftmax:
                # pass  # 这是输出通过softmax函数给出各种情况的概率。此时损失函数是交叉熵损失函数。稍后实现
                loss.append(np.mean(-np.sum(np.multiply(labelBatch, np.log(yp + 1e-1000)), axis=0)))
            else:  # 这是输出只是独立的变量，使用均方根损失函数
                lossList.append(np.mean(np.sqrt(np.sum(np.power(errors, 2), axis=0))))
            self.iteration += 1
            self.BP(errors)
            self.Update()

    @staticmethod
    def Relu(x):
        return x if x > 0 else 0.0

    @staticmethod
    def ReluDiff(x):
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def Identity(x):
        return x

    @staticmethod
    def IdentityDiff(x):
        return 1.0

    @staticmethod
    def softmax(x):
        x[x > 1e2] = 1e2
        ep = np.power(np.e, x)
        return ep / np.sum(ep, axis=0)


if __name__ == '__main__':
    filename = nolinearRegressionDataDir + "数据集.xls"
    xdataOriginal = pd.read_excel(filename, sheet_name="Single_Nonlinear", usecols='D')
    xData = np.array(xdataOriginal)
    labelOriginal = pd.read_excel(filename, sheet_name="Single_Nonlinear", usecols='E')
    labels = np.array(labelOriginal)
    myMultyLayerNeuralNetwork = MultyLayerNeuralNetwork(inputShape=1, shape=[16, 1], activators=['relu', 'identity'],
                                                        learningRate=0.15, threshold=0.001,)
    myMultyLayerNeuralNetwork.Fit(xData, labels)
    min = xData.min()
    max = xData.max()
    X = np.linspace(min, max, 100).reshape(-1, 1)
    predict = myMultyLayerNeuralNetwork.Predict(X)
    plt.plot(xData, labels, 'g.')
    plt.plot(X, predict, 'r-')
    plt.show()
