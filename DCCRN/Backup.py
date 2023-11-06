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
from torch.autograd import Variable
from config import win_size, win_shift, fft_num
EPSILON = 1e-15


class TorchSignalToFrames(object):
    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = (sig_len - self.frame_size) // self.frame_shift + 1
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k=0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size]=in_sig[..., start:]
            start = start + self.frame_shift
            end = start + self.frame_size
        return a



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


class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.frame = TorchSignalToFrames(frame_size=self.frame_size, frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().cuda()
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().cuda()
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().cuda()

    def __call__(self, outputs, labels):
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        output_r, output_i = self.get_stftm(outputs)
        label_r, label_i = self.get_stftm(labels)
        loss = (torch.abs(label_r - output_r) + torch.abs(label_i - output_i)).mean()
        return loss

    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        return stft_R, stft_I


def com_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum() + EPSILON
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

def sisdr_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            utt_len = (frame_list[i] - 1) * win_shift + win_size - win_size
            tmp_mask = torch.ones(utt_len, dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)

    esti_filted, label_filted = esti * mask_for_loss, label * mask_for_loss
    s_target = torch.sum(esti_filted * label_filted, dim=-1, keepdim=True) / (torch.sum(label_filted**2, dim=-1,
                                                                            keepdim=True) + EPSILON) * label_filted
    e_noise = esti_filted - s_target
    loss = -10 * torch.log10(torch.sum(s_target**2, dim=-1, keepdim=True) / (torch.sum(e_noise**2, dim=-1, keepdim=True)) + EPSILON)
    return torch.mean(loss)


def snr_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            utt_len = (frame_list[i] - 1) * win_shift + win_size - win_size
            tmp_mask = torch.ones(utt_len, dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)

    esti_filted, label_filted = esti * mask_for_loss, label * mask_for_loss
    esti_noise = label_filted - esti_filted
    label_norm = torch.sum(label_filted ** 2.0, dim=-1, keepdim=True)
    noise_norm = torch.sum(esti_noise ** 2.0, dim=-1, keepdim=True)
    loss = -10 * torch.log10(label_norm / (noise_norm + EPSILON) + EPSILON)
    return torch.mean(loss)

def t_mag_mse_loss(esti, label, frame_mask_list):
    mask_for_loss = frame_mask_list
    loss = (((esti - label) ** 2) * mask_for_loss).sum() / mask_for_loss.sum()
    return loss


def mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
    loss = (((esti - label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum() + EPSILON
    return loss


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