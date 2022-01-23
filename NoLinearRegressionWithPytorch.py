# 基于Pytorch的非线性回归分析
import pandas as pd
from torchvision import datasets,transforms
import torch
from torch import nn
import numpy as np
from torchsummary import summary
from BatchGenerator import BatchGenerator
from ID_DEFINE import *


def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    xValues = Data['x0'].values
    labelValues = Data['y2'].values
    return xValues, labelValues


class NoLinearRegressionNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    def forward(self, x):
        output = self.fc(x)
        return output

if __name__ == "__main__":
    filename = nolinearRegressionDataDir + 'ax2+bx+c.csv'
    xValues, labels = OpenDataFile(filename)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 5
    model = NoLinearRegressionNet(1)
    # summary(model,input_size=(1), batch_size=-1)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_epoches = 10
    train_loader = BatchGenerator([xValues, labels], batch_size, shuffle=True)
    for [xBatch, labelBatch], epoch in train_loader:
        xBatch = torch.tensor(xBatch)
        labelBatch = torch.tensor(labelBatch)
        train_accuracy_total = 0
        train_correct = 0
        train_loss_sum = 0
        model.train()
        xBatch = xBatch.to(device) # 这句话现在好使了，因为xBatch已经是tensor型变量了
        labelBatch = labelBatch.to(device)
        outputs = model(xBatch)
        loss = criterion(outputs,labelBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        _,predicts = torch.max(outputs.data,1)
        train_accuracy_total +=labelBatch.size(0)
        train_correct += (predicts == labelBatch).cpu().sum().item()
    # test_acc = evaluate_accuracy(test_loader,model)
    # print('epoch %d,loss %.5f,train accuracy %.3f,test accuracy %.3f'%(epoch,train_loss_sum/batch_size,train_correct/train_accuracy_total,test_acc))
print("finish training")

