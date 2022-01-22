# 测试mean()函数用法

import numpy as np

a = np.array([[1,2,3],[4,5,6]])
print(a)

b = np.mean(a,axis=0)
print(b)

yp = np.random.random((1,10))
print("yp=",yp)
c = np.log(yp + 1e-1000)
print("c=",c)
labels = np.random.random((1,10))
print("labels=",labels)
d = np.multiply(labels, c)
print('d=',d)
e = np.mean(d,axis=0)
print("e=",e)


a = np.random.random((2,10))-0.5
print("a=",a)
b = np.power(a,2)
print("b=",b)
c = np.sum(b,axis=0)
print("c=",c)
d = np.sqrt(c)
print("d=",d)
e = np.mean(d)
print("e=",e)