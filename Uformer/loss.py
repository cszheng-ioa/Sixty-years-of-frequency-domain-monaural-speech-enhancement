#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from trans import MelTransform 
EPSILON = torch.finfo(torch.float32).eps

def sisnr(x, s, eps=EPSILON):
    """
    Arguments
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
    Return
        sisnr: N tensor
    """
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    
    x_zm = x - torch.mean(x)
    s_zm = s - torch.mean(s)
    t = torch.sum(x_zm * s_zm) * s_zm / (l2norm(s_zm)**2 + eps)
    return 0.0-20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def calloss(output, source):
    loss_total = 0.0
    zerocount = 0
    for i in range(output.shape[0]):
        loss = sisnr(output[i], source[i])
        if torch.mean(source[i]**2) < 1.2e-8:
            loss = 0.0
            zerocount = zerocount + 1
        loss_total = loss_total + loss 
    loss_total = loss_total / (output.shape[0] - zerocount)
    return loss_total, loss_total, loss_total


def calloss_cplxmse(output, source):
    # B 2 F T
    loss = 0
    output_real, output_imag = output[:,0], output[:,1]
    source_real, source_imag = source[:,0], source[:,1]
    for i in range(output.shape[0]):
        loss_real = F.mse_loss(output_real[i], source_real[i], reduction='sum')
        loss_real = loss_real / output_real.shape[-2]
        loss_imag = F.mse_loss(output_imag[i], source_imag[i], reduction='sum')
        loss_imag = loss_imag / output_imag.shape[-2]
        loss = loss + loss_real + loss_imag
    return loss / output.shape[0] / 2, loss / output.shape[0] / 2, loss / output.shape[0] / 2
    
def calloss_magmse(output, source):
    output_mag = torch.sqrt(torch.clamp(output[:,0]**2 + output[:,1]**2, EPSILON))
    source_mag = torch.sqrt(torch.clamp(source[:,0]**2 + source[:,1]**2, EPSILON))
    loss = 0
    for i in range(output.shape[0]):
        loss_mag = F.mse_loss(output_mag[i], source_mag[i], reduction='sum')
        loss_mag = loss_mag / output_mag.shape[-2]
        loss = loss + loss_mag
    return loss / output.shape[0], loss / output.shape[0], loss / output.shape[0]

def calloss_cplxmse_subband(output, source):
    loss = 0
    output = output[:, :, 1:]
    source = source[:, :, 1:]
    output_real, output_imag = output[:,0], output[:,1]
    source_real, source_imag = source[:,0], source[:,1] # N F T
    output_real = output_real.chunk(4, -2)
    output_imag = output_imag.chunk(4, -2)
    source_real = source_real.chunk(4, -2)
    source_imag = source_imag.chunk(4, -2)
    output_real = torch.stack(output_real, -1) #N F' T 4
    output_imag = torch.stack(output_imag, -1)
    source_real = torch.stack(source_real, -1)
    source_imag = torch.stack(source_imag, -1)
    # weight = [0.4, 0.2, 0.15, 0.1, 0.06, 0.04, 0.03, 0.02]
    weight = [1.5, 1.2, 0.8, 0.5]
    for i in range(output_real.shape[0]):
        for j in range(output_real.shape[-1]):
            loss_real = F.mse_loss(output_real[i, :, :, j], source_real[i, :, :, j], reduction='sum')
            loss_real = weight[j] * loss_real
            loss_imag = F.mse_loss(output_imag[i, :, :, j], source_imag[i, :, :, j], reduction='sum')
            loss_imag = weight[j] * loss_imag
            loss = loss + loss_real + loss_imag
    return loss / output.shape[0]  / output.shape[2] / 2, loss / output.shape[0]  / output.shape[2] / 2, loss / output.shape[0]  / output.shape[2] / 2
    
def calloss_magmse_subband(output, source):
    output_mag = torch.sqrt(torch.clamp(output[:,0]**2 + output[:,1]**2, EPSILON))
    source_mag = torch.sqrt(torch.clamp(source[:,0]**2 + source[:,1]**2, EPSILON))
    # output_mag = output
    # source_mag = source
    loss = 0
    output_mag = output_mag[:, 1:] # N F T
    source_mag = source_mag[:, 1:]
    # weight = [0.4, 0.2, 0.15, 0.1, 0.06, 0.04, 0.03, 0.02]
    weight = [1.5, 1.2, 0.8, 0.5]
    output_mag = output_mag.chunk(4, -2)
    output_mag = torch.stack(output_mag, -1)
    source_mag = source_mag.chunk(4, -2)
    source_mag = torch.stack(source_mag, -1) #N F' T 4
    for i in range(output_mag.shape[0]):
        for j in range(output_mag.shape[-1]):
            loss_mag = F.mse_loss(output_mag[i, :, :, j], source_mag[i, :, :, j], reduction='sum')
            loss_mag = weight[j] * loss_mag
            loss = loss + loss_mag
    return loss / output.shape[0] / output_mag.shape[2], loss / output.shape[0] / output_mag.shape[2], loss / output.shape[0] / output_mag.shape[2]

def calloss_fbankmse_subband(output, source):
    mel = MelTransform(960, sr=48000, num_mels=128)
    output_mag = torch.sqrt(torch.clamp(output[:,0]**2 + output[:,1]**2, EPSILON))
    source_mag = torch.sqrt(torch.clamp(source[:,0]**2 + source[:,1]**2, EPSILON))
    output_mag = output_mag.transpose(1, 2) # N T F
    source_mag = source_mag.transpose(1, 2)    
    output_mag = mel(output_mag)
    source_mag = mel(source_mag)
    output_mag = output_mag.chunk(8, -1)
    output_mag = torch.stack(output_mag, -1)
    source_mag = source_mag.chunk(8, -1)
    source_mag = torch.stack(source_mag, -1) #N T F' 8
    weight = [0.4, 0.2, 0.15, 0.1, 0.06, 0.04, 0.03, 0.02]
    loss = 0
    for i in range(output_mag.shape[0]):
        for j in range(output_mag.shape[-1]):
            loss_mag = F.mse_loss(output_mag[i, :, :, j], source_mag[i, :, :, j], reduction='sum')
            loss_mag = weight[j] * loss_mag
            loss = loss + loss_mag
    return loss / output.shape[0] / output_mag.shape[2], loss / output.shape[0] / output_mag.shape[2], loss / output.shape[0] / output_mag.shape[2]

def calloss_timemae(output, source):
    loss = 0
    for i in range(output.shape[0]):
        loss_time = F.l1_loss(output[i], source[i], reduction='sum')
        loss = loss + loss_time
    return loss / output.shape[0], loss / output.shape[0], loss / output.shape[0]

def calloss_bce(output, source):
    loss = 0
    # for i in range(output.shape[0]):
    #     loss_bce = F.binary_cross_entropy(F.sigmoid(output), source, reduction='sum')
    #     loss = loss + loss_bce
    loss = F.binary_cross_entropy(output, source, reduction='sum')
    return loss/output.shape[0]/output.shape[1], loss/output.shape[0]/output.shape[1], loss/output.shape[0]/output.shape[1]

def calacc(output, source):
    a = torch.zeros(output.shape, device = output.device)
    b = torch.ones(output.shape, device = output.device)
    output = torch.where(output<=0.5, a, b)
    error = int(torch.sum(torch.abs(output-source)).item())
    total = output.shape[0] * output.shape[1] * output.shape[2]
    acc = (total-error) / total
    return acc, acc, acc

def calloss_asrenc(output, source):
    # n t f
    loss = F.mse_loss(output, source, reduction='sum')
    loss = loss / output.shape[0]/output.shape[-2]
    return loss, loss, loss