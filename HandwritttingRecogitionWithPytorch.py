import torch
from torch import optim
from tqdm import tqdm   #python的进度条模块
import numpy as np
from torch import nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
# %matplotlib inline
mnist = MNIST(root='D:/WorkSpace/DataSet', train=True, download=True)
def MyImageShow():
    for i, j in enumerate(np.random.randint(0, len(mnist),(10,))):
        data, label = mnist[j]
        plt.subplot(2, 5, i+1)
        plt.imshow(data)
    plt.show()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputLayer = nn.Sequential(nn.Linear(28*28,256), nn.ReLU(), nn.Dropout(0.2))
        self.hiddenLayer = nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.2))
        self.outputLayer = nn.Sequential(nn.Linear(256,10))

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.hiddenLayer(x)
        x = self.outputLayer(x)
        return x

def accuracy(pred, target):
    pred_label = torch.argmax(pred,1)
    correct = sum(pred_label == target).to(torch.float)
    #acc = correct / float(len(pred))
    return correct, len(pred)


if __name__ == "__main__":
    MyImageShow()
    myMLP = MLP()
    print(myMLP)