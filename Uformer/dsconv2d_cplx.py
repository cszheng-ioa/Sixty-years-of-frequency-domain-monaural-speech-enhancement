#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv2d_cplx import ComplexConv2d_Encoder
EPSILON = torch.finfo(torch.float32).eps

class DSConv2d(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels,
                 conv_channels,
                 dilation1,
                 dilation2,
                 kernel_size=3,
                 causal=False):
        super(DSConv2d, self).__init__()
        # 1x1 conv
        self.conv1x1 = ComplexConv2d_Encoder(in_channels, conv_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0,0))
        self.prelu = nn.PReLU()
        self.layernorm_conv1 = nn.LayerNorm(in_channels)
        dconv_pad1 = (dilation1 * (kernel_size - 1)) // 2 if not causal else (
            dilation1 * (kernel_size - 1))
        dconv_pad2 = (dilation2 * (kernel_size - 1)) // 2 if not causal else (
            dilation2 * (kernel_size - 1))
        # depthwise conv
        self.dconv1 = ComplexConv2d_Encoder(conv_channels, conv_channels, kernel_size=(3, kernel_size), stride=(1, 1), padding=(1,dconv_pad1), dilation = (1,dilation1))
        self.dconv2 = ComplexConv2d_Encoder(conv_channels, conv_channels, kernel_size=(3, kernel_size), stride=(1, 1), padding=(1,dconv_pad2), dilation = (1,dilation2))
        self.layernorm_conv2 = nn.LayerNorm(conv_channels)
        # 1x1 conv cross channel
        self.sconv = ComplexConv2d_Encoder(conv_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0,0))
        # different padding way
        self.causal = causal
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        # N C F T 2
        y = self.layernorm_conv1(x.transpose(1,4)).transpose(1,4)

        y = self.conv1x1(y)
        y = self.prelu(y)
        
        y1 = self.dconv1(y)
        y2 = self.dconv2(y)
        
        y = y1 * torch.sigmoid(y2)
        y = self.layernorm_conv2(y.transpose(1,4)).transpose(1,4)
        y = y * torch.sigmoid(y)
        y = self.sconv(y)
        y = self.dropout(y)
        y = x + y
        return y

if __name__ == '__main__':
    net = DSConv2d(128, 64, 2, 4)
    inputs = torch.ones([10, 128, 4, 397, 2])
    y = net(inputs)
    print(y.shape)
