#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from linear_real import Real_Linear
from linear_cplx import Complex_Linear



EPSILON = torch.finfo(torch.float32).eps



class T_att(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(T_att, self).__init__()
        self.query = Real_Linear(in_channel, hidden_channel)
        self.key = Real_Linear(in_channel, hidden_channel)
        self.value = Real_Linear(in_channel, hidden_channel)
        self.softmax = nn.Softmax(dim = -1)
        self.hidden_channel = hidden_channel

    def forward(self, q, k, v):
        causal = False
        # NF * T * C
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / self.hidden_channel**0.5
        if causal:
            mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2]), diagonal=0)
            mask = mask.to(energy.device)
            energy = energy * mask
        energy = self.softmax(energy) # NF * T * T
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        
        return weighted_value

class Self_Attention_T(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_T, self).__init__()
        self.T_att1 = T_att(in_channel, hidden_channel)
        self.T_att2 = T_att(in_channel, hidden_channel)
        self.T_att3 = T_att(in_channel, hidden_channel)
        self.T_att4 = T_att(in_channel, hidden_channel)
        self.T_att5 = T_att(in_channel, hidden_channel)
        self.T_att6 = T_att(in_channel, hidden_channel)
        self.T_att7 = T_att(in_channel, hidden_channel)
        self.T_att8 = T_att(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*F, T, C, 2
        x = self.layernorm1(x.transpose(2, 3)).transpose(2, 3)
        real, imag = x[...,0], x[...,1]
        A = self.T_att1(real, real, real)
        B = self.T_att2(real, imag, imag)
        C = self.T_att3(imag, real, imag)
        D = self.T_att4(imag, imag, real)
        E = self.T_att5(real, real, imag)
        F = self.T_att6(real, imag, real)
        G = self.T_att7(imag, real, real)
        H = self.T_att8(imag, imag, imag)
        real_att = A-B-C-D
        imag_att = E+F+G-H
        out = torch.stack([real_att, imag_att], -1)
        out = self.layernorm2(out.transpose(2, 3)).transpose(2, 3)
        return out

class Multihead_Attention_T_Branch(nn.Module):
     def __init__(self, in_channel, hidden_channel, n_heads=1):
         super(Multihead_Attention_T_Branch, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_T(in_channel, hidden_channel) for _ in range(n_heads)] )
         self.transform_linear = Complex_Linear(hidden_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T * 2
        
        N, C, F, T, ri = inputs.shape
        x = inputs.permute(0, 2, 3, 1, 4) # N F T C 2
        x = x.contiguous().view([N*F, T, C, ri])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transform_linear(x)
        outs = outs.contiguous().view([N, F, T, C, ri]) 
        outs = outs.permute(0, 3, 1, 2, 4)
        outs = self.prelu(self.layernorm3(outs.transpose(1, 4)).transpose(1, 4))
        outs = self.dropout(outs)
        outs = outs + inputs
        return outs

if __name__ == '__main__':
    net = Multihead_Attention_T_Branch(128, 64)
    inputs = torch.ones([10, 128, 4, 397, 2])
    y = net(inputs)
    print(y.shape)
