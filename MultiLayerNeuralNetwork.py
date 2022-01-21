import numpy as np


def Relu(inputVectors):
    outputVectors = np.array([0 if d < 0 else d for d in inputVectors]).reshape(-1, 1)
    return outputVectors


class MultyLayerNeuralNetwork:
    def __init__(self, inputShape, shape, activators):
        self.activatorFunc = []
        self.activatorFuncDiff = []
        for activator in activators:
            if activator.upper() == 'RELU':
                self.activatorFunc.append(np.vectorize(self.Relu))

    @staticmethod
    def Relu(x):
        return x if x > 0 else 0.0

    @staticmethod
    def ReluDiff(x):
        return 1.0 if x > 0 else 0.0



if __name__ == '__main__':
    myMultyLayerNeuralNetwork = MultyLayerNeuralNetwork(3,3,['relu'])
    inputVectors = np.random.rand(10, 1) - 0.5
    print("input", inputVectors)
    outputVectors = myMultyLayerNeuralNetwork.activatorFunc[0](inputVectors)
    print("output", outputVectors)
