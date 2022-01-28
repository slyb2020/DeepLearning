import torch
from torch import optim
from tqdm import tqdm  # python的进度条模块
import numpy as np
from torch import nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader


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

    myMLP = MLP()
    print(myMLP)
    Optimizer = optim.SGD(myMLP.parameters(), lr=1e-2, momentum=0.9)
    Loss = nn.CrossEntropyLoss()
    minLoss = 0
    ACC = {'train': [], "val": []}
    LOSS = {'train': [], 'val': []}
    for epoch in tqdm(range(10)):
        # myMLP.eval()  #设置为验证模式
        myMLP.train()
        numTr, denumerTr, lossSum = 0., 0., 0.,
        for data, target in trainLoader:
            Optimizer.zero_grad()  # 梯度初始化为0
            output = myMLP(data)
            loss = Loss(output, target)
            lossSum += loss
            loss.backward()
            Optimizer.step()
            num, denum = accuracy(output, target)
            numTr += num
            denumerTr += denum
        # LOSS['train'].append(loss)
