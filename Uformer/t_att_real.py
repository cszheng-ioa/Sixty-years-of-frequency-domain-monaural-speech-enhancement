#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from linear_real import Real_Linear

EPSILON = torch.finfo(torch.float32).eps



class T_att_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(T_att_real, self).__init__()
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
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / 16**0.5
        if causal:
            mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2]), diagonal=0)
            mask = mask.to(energy.device)
            energy = energy * mask
        energy = self.softmax(energy) # NF * T * T
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        
        return weighted_value

class Self_Attention_T_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_T_real, self).__init__()
        self.T_att = T_att_real(in_channel, hidden_channel)

        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*F, T, C
        out = self.layernorm1(x)
        out = self.T_att(out, out, out)
        out = self.layernorm2(out)
        return out

class Multihead_Attention_T_Branch_real(nn.Module):
     def __init__(self, in_channel, hidden_channel, n_heads=1):
         super(Multihead_Attention_T_Branch_real, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_T_real(in_channel, hidden_channel) for _ in range(n_heads)] )
         self.transform_linear = Real_Linear(hidden_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T
        
        N, C, F, T = inputs.shape
        x = inputs.permute(0, 2, 3, 1) # N F T C
        x = x.contiguous().view([N*F, T, C])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transform_linear(x)
        outs = outs.contiguous().view([N, F, T, C]) 
        outs = self.prelu(self.layernorm3(outs))
        outs = self.dropout(outs)
        outs = outs.permute(0, 3, 1, 2)
        outs = outs + inputs
        return outs



if __name__ == '__main__':
    net = Multihead_Attention_T_Branch_real(128, 64)
    inputs = torch.ones([10, 128, 4, 397])
    y = net(inputs)
    print(y.shape)
