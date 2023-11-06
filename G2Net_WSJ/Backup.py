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

def feat_preprocess(data, type):
    """
    Adopt the compression toward the data
    :param data: (B,2,T,F)
    :param type: (B,2,T,F)
    :return:
    """
    if data.dim() == 4:   # (B, 2, T, F)
        mag, phase = torch.norm(data, dim=1), torch.atan2(data[:,-1,...], data[:,0,...])
        if type == 'sqrt':
            mag = mag**0.5
            return torch.stack((mag*torch.cos(phase), mag*torch.sin(phase)), dim=1)
        elif type == 'cubic':
            mag = mag**0.3
            return torch.stack((mag*torch.cos(phase), mag*torch.sin(phase)), dim=1)
        else:
            return data
    elif data.dim() == 5:   # (B, M, 2, T, F)
        mag, phase = torch.norm(data, dim=2), torch.atan2(data[:,:, -1, ...], data[:,:, 0, ...])
        if type == 'sqrt':
            mag = mag ** 0.5
            return torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=2)
        elif type == 'cubic':
            mag = mag ** 0.3
            return torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=2)
        else:
            return data

def feat_handler(data, type):
    """
    Adopt the data handle accoring to the loss type
    :param data: (B,2,T,F)
    :param type: (B,2,T,F)/(B,T,F)
    :return:
    """
    if type == 'mag':
        return torch.norm(data, dim=1)
    elif type == 'com':
        return data


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

def com_mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-2]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    return 0.5*(loss1 + loss2), (0.5*(loss1 + loss2)).item()

def stagewise_regularize_com_mag_mse_loss(esti_list, mag_list, label, frame_list):
    alpha_list = [0.1 for _ in range(len(esti_list))]
    alpha_list[-1] = 1
    mask_for_loss = []
    utt_num = label.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], label.size()[-2]), dtype=label.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
        mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss1, loss2, loss_regu = 0., 0., 0.
    mag_label = torch.norm(label, dim=1)
    for i in range(len(esti_list)):
        curr_esti = esti_list[i]
        curr_regu = mag_list[i]
        mag_esti = torch.norm(curr_esti, dim=1)
        loss1 = loss1 + alpha_list[i] * (((curr_esti - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
        loss2 = loss2 + alpha_list[i] * (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
        loss_regu = loss_regu + alpha_list[i] * (((curr_regu - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2), (0.5 * (loss1 + loss2)).item(), loss_regu


def stagewise_com_mag_mse_loss(esti_list, label, frame_list):
    alpha_list = [0.1 for _ in range(len(esti_list))]
    alpha_list[-1] = 1
    mask_for_loss = []
    utt_num = label.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], label.size()[-2]), dtype=label.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
        mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss1, loss2 = 0., 0.
    mag_label = torch.norm(label, dim=1)
    for i in range(len(esti_list)):
        curr_esti = esti_list[i]
        mag_esti = torch.norm(curr_esti, dim=1)
        loss1 = loss1 + alpha_list[i] * (((curr_esti - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
        loss2 = loss2 + alpha_list[i] * (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    return 0.5*loss1 + 0.5*loss2, (0.5*loss1 + 0.5*loss2).item()


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num