# -*- encoding: utf-8 -*- 

# @Time    :  2019/5/15 17:55

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set()

hidden_size = 2
num_layers = 4
input_size = 8
batch = 16
seq = 32

lstm = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# input size 表示输入 x(t) 的特征维度
# hidden_size 表示输出 h(t) 的特征维度
# num_layers 表示网络层数


# 超参设置

time_step = 20  # RNN时间步长
batch_size = 1  # RNN输入尺寸
lr = 0.02  # 初始学习率
n_epochs = 100  # 训练回数


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        out, h = self.lstm(x, h)
        prediction = self.out(out)
        return prediction, h


lstm = LSTM()

optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
loss_func = nn.MSELoss()
h_state = None  # 初始化隐藏层

plt.figure()
plt.ion()

for step in range(n_epochs):
    start, end = step * np.pi, (step + 1) * np.pi  # 时间跨度
    # 使用Sin函数预测Cos函数
    steps = np.linspace(start, end, time_step, dtype=np.float32, endpoint=False)

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # print(x_np.shape, y_np.shape) (20,) (20,)

    x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])  # 尺寸大小为(time_step, batch, input_size)
    y = torch.from_numpy(y_np[:, np.newaxis, np.newaxis])

    # print(x.shape, y.shape) torch.Size([20, 1, 1]) torch.Size([20, 1, 1])

    prediction, h_state = lstm(x, h_state)

    # print(type(h_state), type(h_state[0]), type(h_state[0].detach()))

    # <class 'tuple'> < class 'torch.Tensor' > < class 'torch.Tensor' >

    # print(h_state[0].size()) torch.Size([1, 1, 32])

    h_state = (h_state[0].detach(), h_state[1].detach())

    # print(type(h_state)) <class 'tuple'>

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 绘制中间结果

    # plt.cla()
    plt.plot(steps, y_np.flatten(), 'r-', linestyle='--')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.2)

plt.ioff()
plt.show()
