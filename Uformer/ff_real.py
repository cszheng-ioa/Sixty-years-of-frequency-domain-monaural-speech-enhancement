#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_real import Real_Linear
EPSILON = torch.finfo(torch.float32).eps

class FF_Real(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim):
        super(FF_Real, self).__init__()
        self.layernorm_linear = nn.LayerNorm(in_dim)
        self.linear1 = Real_Linear(in_dim, hidden_dim)
        self.linear2 = Real_Linear(hidden_dim, in_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # N C F T
        y = self.layernorm_linear(x.transpose(1, 3))
        y = self.linear1(y)
        y = self.prelu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = y.transpose(1, 3)
        y = y*0.5 + x
        return y

if __name__ == '__main__':
    net = FF_Real(128, 64)
    inputs = torch.ones([10, 128, 4, 397])
    y = net(inputs)
    print(y.shape)
