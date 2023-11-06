#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = torch.finfo(torch.float32).eps

class Complex_Linear(nn.Module):

    def __init__(
                    self,
                    in_dim,
                    out_dim
                ):
        super(Complex_Linear, self).__init__()
        self.real_linear = nn.Linear(in_dim, out_dim)
        self.imag_linear = nn.Linear(in_dim, out_dim)

    def forward(self,inputs):
        # N, *, F, 2
        inputs_real, inputs_imag = inputs[...,0], inputs[...,1]
        out_real = self.real_linear(inputs_real) - self.imag_linear(inputs_imag)
        out_imag = self.real_linear(inputs_imag) + self.imag_linear(inputs_real)
        return torch.stack([out_real, out_imag], -1)
