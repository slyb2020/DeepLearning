# Pytorch自动微分测试
import torch


x = torch.arange(9).view(3, 3)   #只有float型元素的tensor可以requires_grad
print(x.requires_grad)

x = torch.rand(3, 3, requires_grad=True)
print(x.requires_grad)
w = torch.ones(3, 3, requires_grad=True)
y = torch.sum(torch.mm(w,x))
print(y)
y.backward()
# print("y", y.grad)  # 只有is_leaf的tensor有grad属性
print("x", x.grad)
print("w", w.grad)

x = torch.rand(3, 3, requires_grad=True)
w = torch.ones(3, 3, requires_grad=True)
print(x)
print(w)
yy = torch.mm(x, w)
detached_yy = yy.detach()  #对于新版的Pytorch，detach似乎没啥用
y = torch.mean(yy)
y.backward()
# print(yy.grad)
print("detached_YY=",detached_yy.grad)
print("x.grad=", x.grad)
print("w.grad=", w.grad)

sigma1 = torch.eye(2) * 0.5
print(sigma1)