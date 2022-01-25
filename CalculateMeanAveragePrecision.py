# 计算分类算法的mAP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ID_DEFINE import *

def OpenDataFile(filename):
    Data = pd.read_csv(filename)
    scores = Data['score'].values
    groundTruthLabels = Data['label'].values
    return scores, groundTruthLabels

def CalculateMeanAveragePrecision(scores, groundTruthLabels):
    Data = np.stack((scores, groundTruthLabels), axis=1)
    Data = Data[(-Data[:,0]).argsort()] #二维数组根据第1列排序, 由于argsort()只能按从小到大的顺序排序，所以这里加了个负号
    pNumber = 0
    for i in groundTruthLabels:
        if i == 1:
            pNumber += 1
    # print("pNumber=",pNumber)
    precision = np.zeros((Data.shape[0], 1))
    recall = np.zeros((Data.shape[0], 1))
    for i in range(Data.shape[0]):
        temp = np.zeros((Data.shape[0],1))
        temp[0:i+1]=1
        temp[i+1:]=0
        TP = 0
        for j in range(Data.shape[0]):
            if (temp[j] == 1) and (Data[j,1] == 1):
                TP += 1
        recall[i] = TP / pNumber
        precision[i] = TP / (i + 1)
    # print("precision=", precision)
    # print("recall=", recall)
    rpList = []
    exR = None
    for i,r in enumerate(recall):
        if exR != r:
            exR = r
            rpList.append([r[0], precision[i,0]])
    rpArray = np.array(rpList).reshape(-1,2)
    # print("rpArray=", rpArray)
    averagePrecision = np.mean(rpArray,axis=0)[1]
    return rpArray,averagePrecision


if __name__ == "__main__":
    filename = othersDir + "mAP.csv"
    scores, groundTruthLabels = OpenDataFile(filename)
    rpArray,AP = CalculateMeanAveragePrecision(scores, groundTruthLabels)
    print("AP=", AP)
    plt.scatter(rpArray[:,0],rpArray[:,1],c='r',marker='*')
    plt.plot(rpArray[:,0],rpArray[:,1],'r-')
    plt.show()