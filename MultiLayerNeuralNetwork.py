import numpy as np
import pandas as pd
from ID_DEFINE import *


from BatchGenerator import BatchGenerator


def Relu(inputVectors):
    outputVectors = np.array([0 if d < 0 else d for d in inputVectors]).reshape(-1, 1)
    return outputVectors


class MultyLayerNeuralNetwork:
    def __init__(self, inputShape, shape, activators, threshold=1e-5, maxEpoch=1000, regularization=True,
                 miniBatchSize=10, momentum=0.9, softmax=False):#这里的shape是除了输入层以外的其它层的shape列表
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

        self.depth = len(shape)
        self.outputs = [np.mat(0)] * (self.depth + 1) #outputs[0]是输入层的输出。
        self.activityLevels = [np.mat(0)] * self.depth

        self.weights = [np.mat(0)] * self.depth
        self.bias = [np.mat(0)] * self.depth

        self.weights[0] = np.mat(np.random.random((shape[0],inputShape))/100.)
        self.bias[0] = np.mat(np.random.random((shape[0],1))/100.)
        for i,idx in enumerate(self.shape[1:]):
            self.weights[i+1] = np.mat(np.random.random((idx, self.shape[i]))/100.)
            self.bias[i+1] = np.mat(np.random.random((idx,1))/100.)

        self.activatorFunc = []
        self.activatorFuncDiff = []
        for activator in activators:
            if activator.upper() == 'RELU':
                self.activatorFunc.append(np.vectorize(self.Relu))
                self.activatorFuncDiff.append(np.vectorize(self.ReluDiff))
            elif activator.upper() == 'IDENTITY':
                self.activatorFunc.append(np.vectorize(self.Identity))
                self.activatorFuncDiff.append(np.vectorize(self.IdentityDiff))

    def Compute(self,inputX):
        result = inputX
        for idx in range(self.depth):
            self.outputs[idx] = result   # outputs[0]是输入层的输出，而idx=0是计算出的result，在idx=1时赋值给outputs[idx]，正好！
            self.activityLevels[idx] = self.weights[idx] * result + self.bias[idx]
            result = self.activatorFunc[idx](self.activityLevels[idx])
        self.outputs[self.depth] = result
        return self.softmax(result) if self.isSoftmax else result


    def Fit(self, inputX, labels):
        batchGenerator = BatchGenerator([inputX, labels], self.miniBatchSize, shuffle=True)
        lossList = []
        exEpoch = 0
        for [xBatch,labelBatch],epoch in batchGenerator:
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
                pass #这是输出通过softmax函数给出各种情况的概率。此时损失函数是交叉熵损失函数。稍后实现
            else:#这是输出只是独立的变量，使用均方根损失函数
                lossList.append(np.mean(np.sqrt(np.sum(np.power(errors,2),axis=0))))

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
    myMultyLayerNeuralNetwork = MultyLayerNeuralNetwork(inputShape=1, shape=[2,1], activators=['relu', 'identity'])
    myMultyLayerNeuralNetwork.Fit(xData,labels)
