import torch
from torch import optim
from tqdm import tqdm  # python的进度条模块
import numpy as np
from torch import nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from ID_DEFINE import *


# %matplotlib inline
def MyImageShow():
    mnist = MNIST(root='D:/WorkSpace/DataSet', train=True, download=True)
    for i, j in enumerate(np.random.randint(0, len(mnist), (10,))):
        data, label = mnist[j]
        plt.subplot(2, 5, i + 1)
        plt.imshow(data)
    plt.show()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputLayer = nn.Sequential(nn.Linear(28 * 28, 256), nn.ReLU(), nn.Dropout(0.2))
        self.hiddenLayer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
        self.outputLayer = nn.Sequential(nn.Linear(256, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.inputLayer(x)
        x = self.hiddenLayer(x)
        x = self.outputLayer(x)
        return x


def accuracy(pred, target):
    pred_label = torch.argmax(pred, 1)
    correct = sum(pred_label == target).to(torch.float)
    # acc = correct / float(len(pred))
    return correct, len(pred)


if __name__ == "__main__":
    # MyImageShow()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnistTrain = MNIST(root='D:/WorkSpace/DataSet', train=True, download=True, transform=trans)
    mnistVal = MNIST(root="D:/WorkSpace/DataSet", train=False, download=True, transform=trans)
    trainLoader = DataLoader(mnistTrain, batch_size=16, shuffle=True, num_workers=2)
    valLoader = DataLoader(mnistVal, batch_size=16, shuffle=True, num_workers=2)
    modelFileName = modelDir + 'MlP.model'
    isExists = os.path.exists(modelFileName)
    if isExists:
        myMLP = torch.load(modelFileName)
        myMLP.eval()

    else:
        myMLP = MLP()
        print(myMLP)
        Optimizer = optim.SGD(myMLP.parameters(), lr=1e-2, momentum=0.9)
        Loss = nn.CrossEntropyLoss()
        minLoss = 0
        ACC = {'train': [], "val": []}
        LOSS = {'train': [], 'val': []}
        for epoch in tqdm(range(10)):
            myMLP.eval()  # 设置为验证模式
            numberVerify, denumberVerify, lossTr = 0.0, 0.0, 0.0
            with torch.no_grad():
                for data, target, in valLoader:
                    output = myMLP(data)
                    loss = Loss(output, target)
                    lossTr += loss
                    number, denumber = accuracy(output, target)
                    numberVerify += number
                    denumberVerify += denumber
            myMLP.train()
            numberTrain, denumberTrain, lossVerify = 0., 0., 0.,
            for data, target in trainLoader:
                Optimizer.zero_grad()  # 梯度初始化为0
                output = myMLP(data)
                loss = Loss(output, target)
                lossVerify += loss
                loss.backward()
                Optimizer.step()
                number, denumber = accuracy(output, target)
                numberTrain += number
                denumberTrain += denumber
            LOSS['train'].append(lossTr / len(trainLoader))
            LOSS['val'].append(lossVerify / len(valLoader))
            ACC['train'].append(numberTrain / denumberTrain)
            ACC['val'].append(numberVerify / denumberVerify)
        print(LOSS, ACC)
        torch.save(myMLP, modelDir + 'MLP.model')
