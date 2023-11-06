#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = torch.finfo(torch.float32).eps


class Real_Linear(nn.Module):

    def __init__(
                    self,
                    in_dim,
                    out_dim
                ):
        super(Real_Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        # N, *, F
        out = self.linear(inputs)
        return out



