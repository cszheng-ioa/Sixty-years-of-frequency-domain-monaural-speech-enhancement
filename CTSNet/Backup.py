"""
This script is the backup function used to support backup support for the SE system
Author: Andong Li
Time: 2019/05
"""
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import librosa
# import pickle
import json
import os
# import h5py
import numpy as np
from scipy import signal
# import matplotlib.pyplot as plt
# from pystoi.stoi import stoi
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
import scipy.linalg as linalg

EPSILON = 1e-10

def calc_sp(mix, clean, data_type, Win_length, Offset_length):

    n_window= Win_length
    n_overlap = n_window- Offset_length
    c = np.sqrt(np.sum((mix ** 2) )/ len(mix))
    mix = mix / c
    clean = clean / c

    mix_x = librosa.stft(mix,
                     n_fft = n_window,
                     hop_length= n_overlap,
                     win_length= n_window,
                     window= 'hamming').T
    clean_x = librosa.stft(clean,
                           n_fft = n_window,
                           hop_length= n_overlap,
                           win_length= n_window,
                           window= 'hamming').T

    mix_angle = np.angle(mix_x)
    clean_angle = np.angle(clean_x)
    mix_x = np.abs(mix_x)
    clean_x = np.abs(clean_x)

    return data_pack(mix_x, mix_angle), data_pack(clean_x, clean_angle)

class data_pack(object):
    def __init__(self, mag, angle):
        self.mag = mag
        self.angle = angle

def batch_cal_max_frame(file_infos):
    max_frame = 0
    for utter_infos in zip(file_infos):
        file_path = utter_infos[0]
        # read mat file
        mat_file = h5py.File(file_path[0])
        mix_feat = np.transpose(mat_file['mix_feat'])
        max_frame = np.max([max_frame, mix_feat.shape[0]])
    return max_frame

def de_pad(pack):
    """
    clear the zero value in each batch tensor
    Note: return is a numpy format instead of Tensor
    :return:
    """
    mix = pack.mix[0:pack.frame_list,:]
    esti = pack.esti[0:pack.frame_list,:]
    speech = pack.speech[0:pack.frame_list,:]
    return mix, esti, speech


class decode_pack(object):
    def __init__(self, mix, esti, speech, frame_list, c_list):
        self.mix = mix
        self.esti = esti
        self.speech = speech
        self.frame_list = frame_list.astype(np.int32)
        self.c_list = c_list

class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""
    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift
    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class Get_STFT(object):
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR =  np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().cuda()
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().cuda()
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().cuda()
    def __call__(self, x):
        x = self.attain_stft(x)
        return x

    def attain_stft(self, x):
        x = x * self.W
        stft_R = torch.matmul(x, self.DR)
        stft_I = torch.matmul(x, self.DI)
        stftm = torch.stack((stft_R, stft_I), dim=-1)
        return stftm

def mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
    loss = (((esti - label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum() + EPSILON
    return loss

def com_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
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
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)

def num2one_encoder(index, r60_num):
    """
    This function aims to transform the num index into one-encoder format for classification
    :param indx:
    :param r60_num:
    :return:
    """
    num_each_r60 = int(12 / r60_num)
    scale_num = np.int(np.ceil((index - 1) / num_each_r60))
    one_hot_matrix = np.eye(r60_num).astype(np.float)
    return one_hot_matrix[scale_num-1, :]


def summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file)
    return count


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num














