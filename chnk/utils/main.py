# This is a sample Python script.

# Press Schunkft+F10 to execute it or replace it with your code.
# Press Double Schunkft to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
class mmChunkFTFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, H,W):
        ctx.H=H
        ctx.W=W
        return torch.mm(H,W)


    @staticmethod
    def backward(ctx, grad_output):
        W=ctx.W
        H=ctx.H
        W.grad_new=torch.mm(H.T[0:2],grad_output)
        return torch.mm(grad_output,W.T),None


class MMCHUNKFT(nn.Module):
    def forward(self, H, W):
        return mmChunkFTFunction.apply(H,W)

torch.manual_seed(1)
criterion = nn.MSELoss()
# Press the green button in the gutter to run the script.

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.params = nn.Parameter(torch.randn(10, 1))
        self.mm_chunkft=MMCHUNKFT()

    def forward(self, x):
        return self.mm_chunkft(x,self.params)



# 生成随机数据
np.random.seed(42)
num_samples = 20
X = np.random.rand(num_samples, 10)  # 生成随机的x值
noise = np.random.normal(0, 0.1, size=(num_samples, 1))  # 添加一些随机噪声
y = 2 * X[:,0].reshape(num_samples,1) + 1 + noise  # 线性关系 y = 2*x + 1 + noise

# 将数据转换为PyTorch的张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 实例化模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    a=model.params.grad_new
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

