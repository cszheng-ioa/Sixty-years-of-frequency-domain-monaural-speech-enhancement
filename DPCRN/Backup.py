"""
This script is the backup function used to support backup support for the SE system
Author: Andong Li
Time: 2019/05
"""
import torch
import torch.nn as nn
import librosa
import pickle
import json
import os
import h5py
import numpy as np
from scipy import signal
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
import scipy.linalg as linalg
EPSILON = 1e-12

def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets(list): networks
        requires_grad(bool): True or False
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def fusion_snr_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
    esti, label = esti*mask_for_loss, label*mask_for_loss

    # cal SI-SNR
    s_t = label*torch.sum(esti*label, dim=-1, keepdim=True)/(torch.sum(label**2, dim=-1, keepdim=True)+EPSILON)
    e_n = esti - s_t
    loss1 = -10*torch.log10(torch.sum(s_t**2, dim=-1) / (torch.sum(e_n**2, dim=-1)+EPSILON)+EPSILON).mean()
    # cal SV-SNR
    loss2 = -10*torch.log10(torch.sum(label**2, dim=1) / torch.sum((esti-label)**2, dim=1)+EPSILON).mean()
    return 0.5*(loss1+loss2)


def com_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    return loss

def mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
    loss = (((esti - label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    return loss


def com_mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    loss2 = (((esti - label) * com_mask_for_loss) ** 2.0).sum() / com_mask_for_loss.sum()
    return 0.5* (loss1 + loss2)


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num