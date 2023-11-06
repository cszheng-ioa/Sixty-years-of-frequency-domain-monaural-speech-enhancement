#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import os
import sys


EPSILON = torch.finfo(torch.float32).eps


def fusion(cplx, mag):
    cplx_mag = torch.sqrt(torch.clamp(cplx[...,0]**2+cplx[...,1]**2, EPSILON))
    mag_out = mag + torch.sigmoid(cplx_mag)
    cplx_real = cplx[...,0] + torch.sigmoid(mag)
    cplx_imag = cplx[...,1] + torch.sigmoid(mag)
    cplx_out = torch.stack([cplx_real, cplx_imag], -1)
    return cplx_out, mag_out

def fusion_magpha(cplx, mag):
    cplx_mag = torch.sqrt(torch.clamp(cplx[...,0]**2+cplx[...,1]**2, EPSILON))
    cplx_pha = torch.atan2(cplx[...,1]+EPSILON, cplx[...,0])
    mag_out = mag + cplx_mag
    cplx_real = cplx[...,0] + mag * torch.cos(cplx_pha)
    cplx_imag = cplx[...,1] + mag * torch.sin(cplx_pha)
    cplx_out = torch.stack([cplx_real, cplx_imag], -1)
    return cplx_out, mag_out