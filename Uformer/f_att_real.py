#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from linear_real import Real_Linear


EPSILON = torch.finfo(torch.float32).eps



class F_att_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(F_att_real, self).__init__()
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
        # output = []
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / self.hidden_channel**0.5
        energy = self.softmax(energy) # NT * F * F
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])

        return weighted_value

class Self_Attention_F_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_F_real, self).__init__()
        self.F_att = F_att_real(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*T, F, C
        out = self.layernorm1(x)
        out = self.F_att(out, out, out)
        out = self.layernorm2(out)
        return out

class Multihead_Attention_F_Branch_real(nn.Module):
     def __init__(self, in_channel, hidden_channel, n_heads=1):
         super(Multihead_Attention_F_Branch_real, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_F_real(in_channel, hidden_channel) for _ in range(n_heads)] )
         self.transform_linear = Real_Linear(hidden_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T 
        
        N, C, F, T = inputs.shape
        x = inputs.permute(0, 3, 2, 1) # N T F C
        x = x.contiguous().view([N*T, F, C])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)

        out = self.transform_linear(x)
        
        out = out.contiguous().view([N, T, F, C]) 
        out = out.permute(0, 3, 2, 1)
        out = self.prelu(self.layernorm3(out.transpose(1, 3)).transpose(1, 3))
        out = self.dropout(out)
        out = out + inputs
        return out

if __name__ == '__main__':
    net = Multihead_Attention_F_Branch_real(128, 64)
    inputs = torch.ones([10, 128, 4, 397])
    y = net(inputs)
    print(y.shape)
