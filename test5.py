#测试np.mat的运算规则
import numpy as np

# inputX = np.random.random((10,3))
# weight = np.random.random((8,3)).T
# bias = np.random.random((8,1)).T
# print(inputX,weight)
# output = np.dot(inputX,weight)+bias
# output = inputX * weight+bias
# print(output)

inputX = np.mat(np.random.random((10,3)))
weight = np.mat(np.random.random((8,3)))
bias = np.mat(np.random.random((8,1)))

output = weight * inputX.T + bias
print(output)
