#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = torch.finfo(torch.float32).eps


class RealConv2d_Encoder(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    dilation=(1,1),
                    groups=1,
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(RealConv2d_Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)

    def forward(self, inputs):
        # inputs : N C F T
        out = self.conv(inputs)
        out = out[...,:inputs.shape[-1]]
        return out

class RealConv2d_Decoder(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    output_padding=(0,0),
                    dilation=(1,1),
                    groups=1,
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(RealConv2d_Decoder, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, dilation = dilation, groups = groups)

    def forward(self,inputs):
        # inputs : N C F T 2

        out = self.conv(inputs)
        out = out[...,:inputs.shape[-1]]
        return out


