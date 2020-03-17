# -*- encoding: utf-8 -*-

# @Time    :  2019/5/14 20:36

import torch
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision

batch_size = 32
epochs = 64

train_data = torchvision.datasets.MNIST(
    root='../data/mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
)

test_data = torchvision.datasets.MNIST(
    root='../data/mnist', train=False
)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

test_x = test_data.data
# torch.Size([10000, 28, 28])

test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor)
# torch.Size([10000, 1, 28, 28])

test_x = test_x[: batch_size * 100] / 255

test_y = test_data.targets[: batch_size * 100]


# print(test_x.shape, test_y.shape) torch.Size([3200, 1, 28, 28]) torch.Size([3200])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2  # padding=(kernel_size-1)/2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


cnn = CNN()

opt = optim.Adam(cnn.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()


for epoch in range(epochs):
    for step, (x, y) in enumerate(train_loader):

        # x.size() torch.Size([batch_size, 1, 28, 28])

        # y.size() torch.Size([batch_size])

        out = cnn(x)

        loss = criterion(out, y)

        opt.zero_grad()

        loss.backward()

        opt.step()

        # test

        out = cnn(test_x)

        out_num = out.data.numpy()
        y_num = test_y.numpy()

        out_num = out_num.argmax(axis=1)

        acc = np.sum(out_num == y_num) / len(y_num)

        print('Epoch {} / {} - acc : {} - loss : {}'.format(epoch +1 , step + 1, acc, loss.detach().numpy()))


