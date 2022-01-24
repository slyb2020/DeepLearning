import torch

a = torch.rand(2, 2, dtype=float)  # 元素类型默认是torch.float32
print(a, a.type, a.dtype, a.is_leaf, a.grad, a.requires_grad)
b = torch.rand(2, 2)
print(b)
c = a * b  # tensor直接乘，就是对应元素相乘，要求乘数和被乘数的维度必须完全一样，或是常数
d = a * 2
print(c, c.is_leaf, c.grad, c.requires_grad)
print(d)

a = torch.zeros(3, 5, dtype=torch.long)
print(a)

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8)
a = a.new_ones(2, 3)
print(a, a.dtype)

b = torch.ones(2, 3)
print("b=", b)

c = b.new_ones(3, 3)
print("c=", c, c.shape, c.shape[0]) #tensor的shape是变量，有多个维度
print(c.size())  #tensor的size是函数，不带参数

# 张量相加：
a = torch.ones(3, 5)
b = torch.ones(3, 5)
c = a + b
print(c)

c = torch.add(a, b)
print(c)

torch.add(a, b, out=c)  # 这里的c一定是一个已经存在的变量，不能是新变量
print("c=", c)

c.add_(c)
print("c=", c)

a = c.view(-1,5,3)
print("a=", a)
a = torch.tensor([1.234],dtype=torch.float64)
print(a.item())

e = b.numpy()  #tensor的numpy也是函数，不带变量
print("e=", e, type(e))

print(torch.cuda.is_available())

device1 = torch.device("cuda")
device2 = torch.device("cpu")
if torch.cuda.is_available():
    a.to(device1)
    b.to(device2)
    a.to("cuda")
    b.to("cpu")
    print("here")
c = a + b
print("cddd=",c)
print("######################################")
a = torch.rand(3,5)
print(a.size())
b = a.numpy()
print(b.shape)

a = torch.rand(5,3)
b = torch.rand(3,2)
c = torch.matmul(a,b)
d = c.data
print("d=",d.requires_grad, d.grad, d.grad_fn, d.is_leaf)


a = torch.rand(3,requires_grad=True)
b = torch.rand(3,requires_grad=True)
print(a)
print(b)
print(a.grad)
print(a.is_leaf)
c = a * b
print("c=",c)
print(c.is_leaf)
d = c.sum()
print("d=",d)
d.backward()
print("a.grad=",a.grad)
print("b.grad=",b.grad)
# print("c.grad=",c.grad)

a = torch.tensor(1.0,requires_grad=True)
z = a**3
z.backward()
print("z.requires_grad=",z.requires_grad)
print("a.grad=",a.grad)


a = torch.rand(2,3,requires_grad=True)
b = torch.rand(3,1,requires_grad=True)
c = torch.matmul(a,b)
print("c=",c)
# c.backward()   # 这句话会报错，因为backward方法智能是标量输出才能调用
d = sum(c)
print("d=",d)
d.backward()
print("a.grad=",a.grad)
print("b.grad=",b.grad)