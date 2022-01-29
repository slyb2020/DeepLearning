# 对基于Pytorch和Keras的MNIST数据集擦偶哦进行测试及对比
import torch
from tensorflow.keras.datasets import mnist
from torchvision.datasets import MNIST
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from ID_DEFINE import *

# print(train_images,train_labels,train_labels.shape)
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize类使用均值和标准差这两个参数将数据转换到[-1,1]之间。
    # 标准差选在0.3081能理解因为所有数据到分布在[-3标准差,3标准差]之间。但是，均值为啥取0.1307？
    # 其实这里的0.1307和0.3081都是事先使用MNIST的训练集数据计算出来的。
    # 另外MNIST数据集中的数据时单色的，所以Channel=1，Noralize里的参数都是1维的
])

mnistTrain = MNIST(root='D:/WorkSpace/DataSet', train=True, download=True, transform=trans)
mnistVal = MNIST(root="D:/WorkSpace/DataSet", train=False, download=True, transform=trans)
trainLoader = DataLoader(mnistTrain, batch_size=16, shuffle=True, num_workers=2)
valLoader = DataLoader(mnistVal, batch_size=16, shuffle=True, num_workers=2)

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

# def softmax(x):
#     s = torch.exp(x)
#     return s / torch.sum(s, dim=1, keepdim=True)# 此处触发了广播机制


if __name__ == "__main__":
    # img, label = next(iter(trainLoader))
    # print("img=",img)
    # print("label=",label)
    # print(img.shape)

    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # img = next(iter(train_images))  #对于一个iterable的对象，可以使用next(iter())来取出其中的一个值
    # print("img=",img)
    # print(img.shape)
    # 可以看出来：keras的load_data类将MNIST数据转换成numpy数组元素取值【0~255】,数组是二维的【高*宽】

    trainPytorch = MNIST(root='D:/WorkSpace/DataSet', train=True, download=True, transform=trans)
    testPytorch = MNIST(root='D:/WorkSpace/DataSet', train=False, download=True, transform=trans)
    # img, label = next(iter(trainPytorch))
    # print(img,label)
    # #torchvision的MNIST类将MNIST数据转换为PIL.Image对象。

    img, label = trainPytorch[1]
    img, label = next(iter(trainPytorch))
    print(img, label)

    modelFileName = modelDir + 'MlP.model'
    isExists = os.path.exists(modelFileName)
    if isExists:
        myMLP = torch.load(modelFileName)
        myMLP.eval()
    print("length=", len(testPytorch))
    length = len(testPytorch)
    idx = np.random.randint(0,length)
    img, label = testPytorch[idx]
    result = myMLP(img)
    # result = softmax(result)
    print("idx=",idx)
    print("result=", result)
    print("label=", label)
