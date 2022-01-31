# 使用Pytorch基于CNN编写FasionMNIST识别程序
# 包含基于TensorBoard的数据可视化代码

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader as Loader
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ID_DEFINE import *


class FashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(self.fc3(x),dim=1)
        return x


def imagesToProbes(net, images):
    """
    以经过训练的网络和图像列表生成预测标签和响应的概率
    """
    output = net(images)
    _, predictTensors = torch.max(output, 1)
    predicts = np.squeeze(predictTensors.numpy())
    return predicts, [F.softmax(el, dim=0)[i].item() for i, el in zip(predicts, output)]


def plotClassesPredicts(net, images, labels):
    """
    使用经过训练的网络以及一批图像和标签生成matplotlib图，改图显示网络的顶部预测及其概率，并与实际标签一起，根据预测是否正确为该信息上色。
    使用上面的“imagesToProbes”函数。
    """
    predicts, probes = imagesToProbes(net, images)
    fig = plt.figure(figsize=(8, 2))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        img = images[idx].squeeze()
        img = img / 2 + 0.5
        npimg = img.numpy()
        ax.imshow(npimg, cmap="Greys")
        ax.set_title("{0},{1:.1f}%\n(label:{2})".format(
            labelClasses[predicts[idx]],
            probes[idx] * 100.0,
            labelClasses[labels[idx]]),
            color=("green" if predicts[idx] == labels[idx].item() else "red"))
    return fig


if __name__ == "__main__":
    batchSize = 4
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainDataset = torchvision.datasets.FashionMNIST('D:/WorkSpace/DataSet', download=True, train=True,
                                                     transform=transformer)
    testDataset = torchvision.datasets.FashionMNIST('D:/WorkSpace/DataSet', download=True, train=False,
                                                    transform=transformer)
    trainLoader = Loader.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = Loader.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    # labelClasses = ("衬衫", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "靴子")
    labelClasses = (
        "T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")
    myFashionMNIST = FashionMNIST()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(myFashionMNIST.parameters(), lr=1, momentum=0.9)

    writer = SummaryWriter(tensorboardDir)
    dataIter = iter(trainLoader)
    images, labels = next(dataIter)

    imgGrid = torchvision.utils.make_grid(images)

    writer.add_image('four_fashion_mnist_images', imgGrid)

    writer.add_graph(myFashionMNIST, images)
    print("Done")
    threshold = 0.0
    runningLoss = 0.0
    print("Start training")
    for epoch in range(5):
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = myFashionMNIST(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            if i % 5000 == 4900:
                print("Epoch: %d Batch: %d Loss is %f" % (epoch + 1, i + 1, runningLoss / 5000))
                writer.add_scalar('training loss',
                                  runningLoss / 5000,
                                  epoch * len(trainLoader) + i)
                writer.add_figure('prediction vs. actuals',
                                  plotClassesPredicts(myFashionMNIST, inputs, labels),
                                  global_step=epoch * len(trainLoader) + 1)
                runningLoss = 0.0
    print("Finish training")
