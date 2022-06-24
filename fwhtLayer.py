import torch
import torch.nn.functional as F
import numpy as np
from utils import *


def min_power(n):
    temp = 1

    while temp < n:
        temp *= 2

    return temp


class SoftThreshold(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.variable = torch.nn.Parameter(torch.from_numpy(np.arange(0,1024)/(10*1024)))
        self.variable.requires_grad=True
        self.T = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
        self.T.requiresGrad = True

    def forward(self, x):
        y = x * self.T
        absolute = abs(y)
        r = F.relu(absolute)
        z = torch.tanh(y)
        return torch.matmul(r, z)


class FwhtLayer(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.num_features = dim
        v = np.zeros((self.num_features, self.num_features))
        for i in range(self.num_features):
            v[:, i] = (i + np.arange(self.num_features)) / (10 * self.num_features)

        self.variable = torch.nn.Parameter(torch.from_numpy(v))
        self.variable.requires_grad = True
        self.T = torch.nn.Parameter(torch.ones((dim,dim), dtype=torch.float32))
        self.T.requiresGrad = True
    def forward(self, x):

        wh_size = x.shape[-2]
        wht_size = min_power(wh_size)
        walsh = torch.from_numpy(get_walsh_matrix(round(np.log2(wht_size)))).float().to("cuda:0")

        print(x.shape)
        x=x.permute(0,3,2,1)
        f_1 = None
        if wht_size > wh_size:
            padder = (0, 0, 0, wht_size - wh_size, 0, wht_size - wh_size, 0, 0)
            f_1 = F.pad(x, pad=padder)

        else:
            f_1 = x
        f_2 = f_1.permute(0, 3, 2, 1)
        f_3 = torch.tensordot(f_2, walsh, dims=([-1], [0]))
        f_4 = f_3.permute(0, 1, 3, 2)
        f_5 = torch.tensordot(f_4, walsh, dims=([-1], [0]))
        ########
        print(f_5.shape)
        print(self.T.shape)
        y = f_5 * self.T
        absolute = abs(y)
        r = F.tanh(absolute-self.variable)
        k = torch.tanh(y)
        z = torch.matmul(r.float(),k.float())
        #######

        f_7 = torch.tensordot(z, walsh, dims=([-1], [0]))
        f_8 = f_7.permute(0, 1, 3, 2)
        f_9 = torch.tensordot(f_8, walsh, dims=([-1], [0]))

        y = f_9.permute(0, 3, 2, 1)

        if wht_size > wh_size:
            y = y[:, :wh_size, :wh_size, :]

        y = y.permute(0, 3, 2, 1)
        print("ysha",y.shape)
        return y

"""
x = (16, 16, 1024)
acti = SoftThreshold()
model = FwhtLayer(min_power(x[-2]))

for p in acti.parameters():
    print(p)
from torchsummary import summary

summary(model, x, device='cpu')"""