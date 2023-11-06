#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = torch.finfo(torch.float32).eps

class ComplexConv2d_Encoder(nn.Module):

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
        super(ComplexConv2d_Encoder, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)

    def forward(self,inputs):
        # inputs : N C F T 2
        inputs_real, inputs_imag = inputs[...,0], inputs[...,1]
        out_real = self.real_conv(inputs_real) - self.imag_conv(inputs_imag)
        out_imag = self.real_conv(inputs_imag) + self.imag_conv(inputs_real)
        out_real = out_real[...,:inputs_real.shape[-1]]
        out_imag = out_imag[...,:inputs_imag.shape[-1]]
        return torch.stack([out_real, out_imag], -1)

class ComplexConv2d_Decoder(nn.Module):

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
        super(ComplexConv2d_Decoder, self).__init__()
        self.real_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, dilation = dilation, groups = groups)
        self.imag_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, dilation = dilation, groups = groups)

    def forward(self,inputs):
        # inputs : N C F T 2
        inputs_real, inputs_imag = inputs[...,0], inputs[...,1]
        out_real = self.real_conv(inputs_real) - self.imag_conv(inputs_imag)
        out_imag = self.real_conv(inputs_imag) + self.imag_conv(inputs_real)
        out_real = out_real[...,:inputs_real.shape[-1]]
        out_imag = out_imag[...,:inputs_imag.shape[-1]]
        return torch.stack([out_real, out_imag], -1)


