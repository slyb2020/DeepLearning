# numpy stack函数的用法：
import numpy as np
# a = np.array([[1,2,3],[4,5,6]])
# print(a)
# b = np.stack(a,axis=0)
# print(b)
# c = np.stack(a,axis=1)
# print(c)


a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.stack((a,b),axis=1)
print("a=",a)
print("b=",b)
print("c=",c)

# a = np.array([1,2,3]).reshape(-1,1)
# b = np.array([4,5,6]).reshape(-1,1)
# d = np.stack((a,b),axis=0)
# print("c=",d)

# axis=0 上下摞
# axis=1 左右摞

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.vstack((a,b))
print("c=",c)