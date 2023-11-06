#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from linear_real import Real_Linear
from linear_cplx import Complex_Linear

EPSILON = torch.finfo(torch.float32).eps



class F_att(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(F_att, self).__init__()
        self.query = Real_Linear(in_channel, hidden_channel)
        self.key = Real_Linear(in_channel, hidden_channel)
        self.value = Real_Linear(in_channel, hidden_channel)
        self.softmax = nn.Softmax(dim = -1)
        self.hidden_channel = hidden_channel

    def forward(self, q, k, v):
        # NT * F * C
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / self.hidden_channel**0.5
        energy = self.softmax(energy) # NT * F * F
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])

        return weighted_value

class Self_Attention_F(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_F, self).__init__()
        self.F_att1 = F_att(in_channel, hidden_channel)
        self.F_att2 = F_att(in_channel, hidden_channel)
        self.F_att3 = F_att(in_channel, hidden_channel)
        self.F_att4 = F_att(in_channel, hidden_channel)
        self.F_att5 = F_att(in_channel, hidden_channel)
        self.F_att6 = F_att(in_channel, hidden_channel)
        self.F_att7 = F_att(in_channel, hidden_channel)
        self.F_att8 = F_att(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*T, F, C, 2
        x = self.layernorm1(x.transpose(2, 3)).transpose(2, 3)
        real, imag = x[...,0], x[...,1]
        A = self.F_att1(real, real, real)
        B = self.F_att2(real, imag, imag)
        C = self.F_att3(imag, real, imag)
        D = self.F_att4(imag, imag, real)
        E = self.F_att5(real, real, imag)
        F = self.F_att6(real, imag, real)
        G = self.F_att7(imag, real, real)
        H = self.F_att8(imag, imag, imag)
        real_att = A-B-C-D
        imag_att = E+F+G-H
        out = torch.stack([real_att, imag_att], -1)
        out = self.layernorm2(out.transpose(2, 3)).transpose(2, 3)
        return out

class Multihead_Attention_F_Branch(nn.Module):
     def __init__(self, in_channel, hidden_channel, n_heads=1):
         super(Multihead_Attention_F_Branch, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_F(in_channel, hidden_channel) for _ in range(n_heads)] )
         self.transform_linear = Complex_Linear(hidden_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T * 2
        N, C, F, T, ri = inputs.shape
        x = inputs.permute(0, 3, 2, 1, 4) # N T F C 2
        x = x.contiguous().view([N*T, F, C, ri])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transform_linear(x)
        outs = outs.contiguous().view([N, T, F, C, ri]) 
        outs = outs.permute(0, 3, 2, 1, 4)
        outs = self.prelu(self.layernorm3(outs.transpose(1, 4)).transpose(1, 4))
        outs = self.dropout(outs)
        outs = outs + inputs
        return outs


if __name__ == '__main__':
    net = Multihead_Attention_F_Branch(128, 64)
    inputs = torch.ones([10, 128, 4, 397, 2])
    y = net(inputs)
    print(y.shape)
