# [np.array(d) for d in allData]用法测试
import numpy as np


def BatchGenerator(allData,bachSize=5,shuffle=True):
    if shuffle:
        print("defore=",allData)
        p = np.random.permutation(dataSize)
        allData = [d[p] for d in allData]
    batchSize=3
    batchCount=0
    while True:
        if batchCount*batchSize+batchSize > dataSize:
            batchCount = 0
            if shuffle:
                p = np.random.permutation(dataSize)
                allData = [d[p] for d in allData]
        start = batchCount * batchSize
        end = start + batchSize
        batchCount += 1
        yield [d[start:end] for d in allData]


if __name__ == "__main__":
    x = np.random.rand(10, 2)
    labels = np.random.rand(10)
    allData = [x, labels]
    dataSize = allData[0].shape[0]
    batchGenerator = BatchGenerator(allData,3)
    for batchData in batchGenerator:
        print("batch Data=",batchData)
