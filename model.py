
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import numpy as np
import scipy.fft

class Thd_relu(nn.Module):
    def __init__(self):
        super(Thd_relu, self).__init__()
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias.data.fill_(0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x + self.bias)
        return x


class Net_Local(nn.Module):
    def __init__(self, D, W):
        super(Net_Local, self).__init__()
        self.Dim_Before = D.shape[0]
        self.Dim_After = W.shape[0]
        self.D = nn.Linear(self.Dim_Before, self.Dim_Before, bias=False)
        self.activation = Thd_relu()
        self.W = nn.Linear(self.Dim_Before, self.Dim_After, bias=False)
        self.D.weight = Parameter(torch.tensor(D))
        self.W.weight = Parameter(torch.tensor(W))

    def forward(self, x):
        x = self.D(x)
        x = self.activation(x)
        x = self.W(x)
        return x


class Net_Classifier(nn.Module):
    def __init__(self, W, b):
        super(Net_Classifier, self).__init__()
        self.Dim_Before = W.shape[1]
        self.Dim_After = W.shape[0]

        self.W = nn.Linear(self.Dim_Before, self.Dim_After, bias=True)
        self.W.weight = Parameter(torch.tensor(W))
        self.W.bias = Parameter(torch.tensor(b))

    def forward(self, x):
        x = self.W(x)
        return x

def Model_Initialization(Dim, depth):
    Network = {}

    for i in range(depth - 2):
        if i == 0:
            if Dim[i + 1] > Dim[i]:
                Network["D" + str(i + 1)] = scipy.fft.dct(np.eye(Dim[i + 1]), axis=0, norm="ortho")[:, :Dim[i]]
            else:
                Network["D" + str(i + 1)] = scipy.fft.dct(np.eye(Dim[i]), axis=0, norm="ortho")[:Dim[i + 1], :]
            Network["alpha" + str(i + 1)] = 0
            Network["W" + str(i + 1)] = np.random.normal(0, 0.005, [Dim[i + 2], Dim[i + 1]])
        elif i == depth - 3:
            Network["D" + str(i + 1)] = scipy.fft.dct(np.eye(Dim[i + 1]), axis=0, norm="ortho")
            Network["alpha" + str(i + 1)] = 0
            Network["W" + str(i + 1)] = np.random.normal(0, 0.005, [Dim[i + 1], Dim[i + 1]])
        else:
            Network["D" + str(i + 1)] = scipy.fft.dct(np.eye(Dim[i + 1]), axis=0, norm="ortho")
            Network["alpha" + str(i + 1)] = 0
            Network["W" + str(i + 1)] = np.random.normal(0, 0.005, [Dim[i + 2], Dim[i + 1]])

    Network["W" + str(depth - 1)] = np.random.normal(0, 0.005, [Dim[-1], Dim[-2]])
    Network["b" + str(depth - 1)] = np.zeros(Dim[-1])

    return Network