import numpy as np
depth = 5
activityLevels = [np.mat(0)] * depth
print(activityLevels)
a = [0]*5
print(a)
b = np.random.random((10,2)) # size=()
print("b=",b)
c = np.random.rand(10,2)  #d0,d1,d2,d3...
print("c=",c)

weight = np.mat(np.random.random((5,4))/100)
print("weitht=",weight)

shape = [3,5,7]
for i,idx in enumerate(shape[1:]):
    print(i,idx)