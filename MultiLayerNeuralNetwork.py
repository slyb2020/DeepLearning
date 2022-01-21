import numpy as np
from BatchGenerator import BatchGenerator


def Relu(inputVectors):
    outputVectors = np.array([0 if d < 0 else d for d in inputVectors]).reshape(-1, 1)
    return outputVectors


class MultyLayerNeuralNetwork:
    def __init__(self, inputShape, shape, activators, threshold=1e-5, maxEpoch=1000, regularization=True,
                 miniBatchSize=50, momentum=0.9):
        if len(activators) != len(shape):
            raise Exception("激活函数数量与神经网络层数不匹配，程序无法运行！请在下次运行程序前确保为每层网络都指定激活函数！")
        self.inputShape = inputShape
        self.shape = shape
        self.treshold = threshold
        self.maxEpoch = maxEpoch
        self.regularization = regularization
        self.miniBatchSize = miniBatchSize
        self.momentum = momentum

        self.depth = len(shape)
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

    def Fit(self, inputX, labels):
        i=0
        batchGenerator = BatchGenerator([inputX, labels], self.miniBatchSize, shuffle=True)
        for [xBatch,labelBatch],epoch in batchGenerator:
            print("xBatch,labelBatch,epoch=",xBatch,labelBatch,epoch)
            i+=1
            if i>=3:
                break

    @staticmethod
    def Relu(x):
        return x if x > 0 else 0.0

    @staticmethod
    def ReluDiff(x):
        return 1.0 if x > 0 else 0.0


if __name__ == '__main__':
    myMultyLayerNeuralNetwork = MultyLayerNeuralNetwork(2, [8,4], ['relu','relu'])
    # print("weights=",myMultyLayerNeuralNetwork.weights)
    # print("bias=",myMultyLayerNeuralNetwork.bias)
    inputVectors = np.random.rand(100, 2) - 0.5
    labels = np.random.random((100,1))
    # print("input", inputVectors)
    # outputVectors = myMultyLayerNeuralNetwork.activatorFunc[0](inputVectors)
    # print("output", outputVectors)
    myMultyLayerNeuralNetwork.Fit(inputVectors,labels)
