# -*- encoding: utf-8 -*-

# @Time    :  2019/5/16 11:59

import torch
from torch import nn
from torch import optim
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt


def load(time_step):
    data = load_boston().target
    xs = []
    ys = []
    for t in range(len(data) - time_step - 1):
        x = data[t: t + time_step]
        y = data[t + time_step]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


time_step = 24

x, y = load(time_step)

x = x[:, :, np.newaxis]
y = y[:, np.newaxis, np.newaxis]


# print(x.shape, y.shape) (481, 24, 1) (481, 1, 1)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1
        )
        self.linear = nn.Linear(32, 1)

    def forward(self, x, h):
        x, h = self.lstm(x, h)
        return self.linear(x), h


lstm = LSTM()
opt = optim.Adam(lstm.parameters(), lr=0.01)
criterion = nn.MSELoss()

x_v = torch.autograd.Variable(torch.Tensor(x))
y_v = torch.autograd.Variable(torch.Tensor(y))
h = None

plt.figure()
plt.ion()

epochs = 500

for epoch in range(epochs):
    o, h = lstm(x_v, h)
    h = (h[0].detach(), h[1].detach())

    o = o[:, -1, :]
    o = o.view(o.size()[0], 1, 1)

    loss = criterion(o, y_v)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print('Epoch {} / {} - loss : {}'.format(epoch, epochs, loss.detach().numpy() / 481))

    plt.cla()
    o_num = o.data.numpy().flatten()
    plt.plot(y.flatten(), label='target', linestyle='--')
    plt.plot(o_num, label='predict')
    plt.draw()
    plt.legend()
    plt.pause(0.001)

plt.ioff()
plt.show()
