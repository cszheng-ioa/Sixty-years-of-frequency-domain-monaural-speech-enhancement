import json
import os
import h5py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import librosa
import random
import soundfile as sf
from Backup import *
from Step2_config import *
EPSILON = 1e-10

class To_Tensor(object):
    def __call__(self, x, type):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return  torch.IntTensor(x)

class TrainDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos= os.path.join(json_dir, 'noisy_trainset_28spk_wav', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start+ batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]

class CvDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos= os.path.join(json_dir, 'noisy_testset_wav', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start+ batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]

# class Step1_TrainDataLoader(object):
#     def __init__(self, data_set, **kw):
#         self.data_loader = DataLoader(dataset=data_set,
#                                       shuffle=1,
#                                       collate_fn=self.collate_fn,
#                                       **kw)
#
#     @staticmethod
#     def collate_fn(batch):
#         feats, labels, noises, frame_mask_list = s1_generate_feats_labels(batch)
#         return S1_BatchInfo(feats, labels, noises, frame_mask_list)
#
#     def get_data_loader(self):
#         return self.data_loader
#
# class Step1_CvDataLoader(object):
#     def __init__(self, data_set, **kw):
#         self.data_loader = DataLoader(dataset=data_set,
#                                       shuffle=1,
#                                       collate_fn=self.collate_fn,
#                                       **kw)
#
#     @staticmethod
#     def collate_fn(batch):
#         feats, labels, noises, frame_mask_list = s1_cv_generate_feats_labels(batch)
#         return S1_BatchInfo(feats, labels, noises, frame_mask_list)
#
#     def get_data_loader(self):
#         return self.data_loader

class Step2_TrainDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)
    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = s2_generate_feats_labels(batch)
        return S2_BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

class Step2_CvDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = s2_cv_generate_feats_labels(batch)
        return S2_BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader


def s2_generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, noise_list, frame_mask_list = [], [], [], []
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        # WSJ
        #        clean_file_name = '%s.wav' % (batch[id].split('_')[0])
        #        mix_file_name = '%s.wav' % (batch[id])
        #        feat_wav, _ = sf.read(os.path.join(file_path, 'train', 'mix', mix_file_name))
        #        label_wav, _ = sf.read(os.path.join(file_path, 'train', 'clean', clean_file_name))
        # VB
        clean_file_name = '%s.wav' % (batch[id])
        mix_file_name = '%s.wav' % (batch[id])
        feat_wav, orig_fs = sf.read(os.path.join(file_path, 'noisy_trainset_28spk_wav', mix_file_name))
        label_wav, _ = sf.read(os.path.join(file_path, 'clean_trainset_28spk_wav', clean_file_name))

        feat_wav = librosa.resample(feat_wav, orig_fs, 16000, fix=True, scale=False)
        label_wav = librosa.resample(label_wav, orig_fs, 16000, fix=True, scale=False)

        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav, label_wav = feat_wav * c, label_wav * c
        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav) - chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]

        frame_num = (len(feat_wav) - win_size + fft_num) // win_shift + 1
#        frame_num = len(feat_wav)

        feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T
        label_x = librosa.stft(label_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

        feat_x, label_x = feat_x[0:frame_num, :], label_x[0:frame_num, :]

        feat_mag, feat_phase = np.abs(feat_x), np.angle(feat_x)
        label_mag, label_phase = np.abs(label_x), np.angle(label_x)
        feat_mag_com, label_mag_com = np.sqrt(feat_mag), np.sqrt(label_mag)
#        feat_mag_com, label_mag_com = feat_mag, label_mag

        feat_real, feat_imag = feat_mag_com * np.cos(feat_phase), feat_mag_com * np.sin(feat_phase)
        label_real, label_imag = label_mag_com * np.cos(label_phase), label_mag_com * np.sin(label_phase)

        feat_x, label_x = to_tensor(np.concatenate((feat_real[:, :, np.newaxis].astype(np.float32),
                                                    feat_imag[:, :, np.newaxis].astype(np.float32)), axis=-1),
                                    'float'), \
                          to_tensor(np.concatenate((label_real[:, :, np.newaxis].astype(np.float32),
                                                    label_imag[:, :, np.newaxis].astype(np.float32)), axis=-1),
                                    'float')  # compressed reverb and clean RI

        feat_list.append(feat_x)
        label_list.append(label_x)
        frame_mask_list.append(frame_num)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)

    feat_list, label_list = feat_list.permute(0, 3, 1, 2).contiguous(), label_list.permute(0, 3, 1, 2).contiguous()

    return feat_list, label_list, frame_mask_list

def s2_cv_generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, noise_list, frame_mask_list = [], [], [], []
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        # WSJ
        #        clean_file_name = '%s.wav' % (batch[id].split('_')[0])
        #        mix_file_name = '%s.wav' % (batch[id])
        #        feat_wav, _ = sf.read(os.path.join(file_path, 'cv', 'mix', mix_file_name))
        #        label_wav, _ = sf.read(os.path.join(file_path, 'cv', 'clean', clean_file_name))
        # VB
        clean_file_name = '%s.wav' % (batch[id])
        mix_file_name = '%s.wav' % (batch[id])
        feat_wav, orig_fs = sf.read(os.path.join(file_path, 'noisy_testset_wav', mix_file_name))
        label_wav, _ = sf.read(os.path.join(file_path, 'clean_testset_wav', clean_file_name))

        feat_wav = librosa.resample(feat_wav, orig_fs, 16000, fix=True, scale=False)
        label_wav = librosa.resample(label_wav, orig_fs, 16000, fix=True, scale=False)

        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav, label_wav = feat_wav * c, label_wav * c
        #        noise_wav = feat_wav - label_wav
        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav) - chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]

        frame_num = (len(feat_wav) - win_size + fft_num) // win_shift + 1
        #        frame_num = len(feat_wav)

        feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T
        label_x = librosa.stft(label_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

        feat_x, label_x= feat_x[0:frame_num, :], label_x[0:frame_num, :]

        feat_mag, feat_phase = np.abs(feat_x), np.angle(feat_x)
        label_mag, label_phase = np.abs(label_x), np.angle(label_x)
        feat_mag_com, label_mag_com = np.sqrt(feat_mag), np.sqrt(label_mag)
#        feat_mag_com, label_mag_com = feat_mag, label_mag

        feat_real, feat_imag = feat_mag_com * np.cos(feat_phase), feat_mag_com * np.sin(feat_phase)
        label_real, label_imag = label_mag_com * np.cos(label_phase), label_mag_com * np.sin(label_phase)

        feat_x, label_x = to_tensor(np.concatenate((feat_real[:, :, np.newaxis].astype(np.float32),
                                                    feat_imag[:, :, np.newaxis].astype(np.float32)), axis=-1),
                                    'float'), \
                          to_tensor(np.concatenate((label_real[:, :, np.newaxis].astype(np.float32),
                                                    label_imag[:, :, np.newaxis].astype(np.float32)), axis=-1),
                                    'float')  # compressed reverb and clean RI

        feat_list.append(feat_x)
        label_list.append(label_x)
        frame_mask_list.append(frame_num)
    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)


    feat_list, label_list = feat_list.permute(0, 3, 1, 2).contiguous(), label_list.permute(0, 3, 1, 2).contiguous()


    return feat_list, label_list, frame_mask_list



class S2_BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list