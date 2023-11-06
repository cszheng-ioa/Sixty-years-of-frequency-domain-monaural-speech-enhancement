import json
import os
import h5py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import soundfile as sf
import librosa
from config import *
EPSILON = 1e-15

class To_Tensor(object):
    def __call__(self, x, type='float'):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return  torch.IntTensor(x)

class TrainDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos = os.path.join(json_dir, 'train', 'files.json')
#        json_pos= os.path.join(json_dir, 'noisy_trainset_28spk_wav', 'files.json')
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
        json_pos = os.path.join(json_dir, 'cv', 'files.json')
#        json_pos= os.path.join(json_dir, 'noisy_testset_wav', 'files.json')
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

class TrainDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

def generate_feats_labels_uncompressed(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        clean_file_name = '%s.wav' % (batch[id].split('_')[0])
        mix_file_name = '%s.wav' % (batch[id])
        feat_wav, _= sf.read(os.path.join(file_path, 'train', 'mix', mix_file_name))
        label_wav, _ = sf.read(os.path.join(file_path, 'train', 'clean', clean_file_name))
        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav, label_wav = to_tensor(feat_wav * c), to_tensor(label_wav * c)
        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav)- chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]

        frame_num = (len(feat_wav)-win_size+win_size) // win_shift + 1
        frame_len = (frame_num-1)*win_shift
        frame_mask_list.append(frame_num)
        feat_list.append(feat_wav)
        label_list.append(label_wav[:frame_len])

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    feat_list = torch.stft(feat_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                           window=torch.hann_window(fft_num)).permute(0,3,2,1)
    label_list = torch.stft(label_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                            window=torch.hann_window(fft_num)).permute(0,3,2,1)
    return feat_list, label_list, frame_mask_list

def generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        #WSJ
        clean_file_name = '%s.wav' % (batch[id].split('_')[0])
        mix_file_name = '%s.wav' % (batch[id])
        feat_wav, orig_fs = sf.read(os.path.join(file_path, 'train', 'mix', mix_file_name))
        label_wav, _ = sf.read(os.path.join(file_path, 'train', 'clean', clean_file_name))
        #VB
#        clean_file_name = '%s.wav' %(batch[id])
#        mix_file_name = '%s.wav'  %(batch[id])
#        feat_wav, orig_fs = sf.read(os.path.join(file_path,  'noisy_trainset_28spk_wav', mix_file_name))
#        label_wav, _ = sf.read(os.path.join(file_path, 'clean_trainset_28spk_wav', clean_file_name))

        feat_wav = librosa.resample(feat_wav, orig_fs, 16000, fix=True, scale=False)
        label_wav = librosa.resample(label_wav, orig_fs, 16000, fix=True, scale=False)

        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav, label_wav = to_tensor(feat_wav * c), to_tensor(label_wav * c)
        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav) - chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]
        frame_num = (len(feat_wav) - win_size + win_size) // win_shift + 1
        frame_len = (frame_num - 1) * win_shift
        frame_mask_list.append(frame_num)

        feat_list.append(feat_wav)
        label_list.append(label_wav)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    #     if len(feat_wav) > chunk_length:
    #         wav_start = random.randint(0, len(feat_wav) - chunk_length)
    #         feat_wav = feat_wav[wav_start:wav_start + chunk_length]
    #         label_wav = label_wav[wav_start:wav_start + chunk_length]
    #
         # frame_num = (len(feat_wav) - win_size+win_size) // win_shift + 1
         # frame_len = (frame_num - 1) * win_shift
         # frame_mask_list.append(frame_num)
    #     feat_list.append(feat_wav)
    #     label_list.append(label_wav[:frame_len])
    #
    # feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    # label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    # feat_list = torch.stft(feat_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
    #                        window=torch.hann_window(fft_num)).permute(0,3,2,1)
    # label_list = torch.stft(label_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
    #                         window=torch.hann_window(fft_num)).permute(0,3,2,1)
    #
    # mag_feat, mag_label = torch.norm(feat_list, dim=1) ** 1.0, torch.norm(label_list, dim=1) ** 1.0
    # phase_feat, phase_label = torch.atan2(feat_list[:, -1, :, :], feat_list[:, 0, :, :]), torch.atan2(label_list[:, -1, :, :], label_list[:, 0, :, :])
    # feat_list, label_list = torch.stack((mag_feat * torch.cos(phase_feat), mag_feat * torch.sin(phase_feat)), dim=1), \
    #                           torch.stack((mag_label * torch.cos(phase_label), mag_label * torch.sin(phase_label)), dim=1)
    return feat_list, label_list, frame_mask_list

def cv_generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    to_tensor = To_Tensor()
    for id in range(len(batch)):
        #WSJ
        clean_file_name = '%s.wav' % (batch[id].split('_')[0])
        mix_file_name = '%s.wav' % (batch[id])
        feat_wav, orig_fs = sf.read(os.path.join(file_path, 'cv', 'mix', mix_file_name))
        label_wav, _ = sf.read(os.path.join(file_path, 'cv', 'clean', clean_file_name))
        #VB
#        clean_file_name = '%s.wav' %(batch[id])
#        mix_file_name = '%s.wav' % (batch[id])
#        feat_wav, orig_fs = sf.read(os.path.join(file_path,  'noisy_testset_wav', mix_file_name))
#        label_wav, _ = sf.read(os.path.join(file_path, 'clean_testset_wav', clean_file_name))

        feat_wav = librosa.resample(feat_wav, orig_fs, 16000, fix=True, scale=False)
        label_wav = librosa.resample(label_wav, orig_fs, 16000, fix=True, scale=False)


        c = np.sqrt(len(feat_wav) / np.sum(feat_wav ** 2.0))
        feat_wav, label_wav = to_tensor(feat_wav * c), to_tensor(label_wav * c)
        if len(feat_wav) > chunk_length:
            wav_start = random.randint(0, len(feat_wav) - chunk_length)
            feat_wav = feat_wav[wav_start:wav_start + chunk_length]
            label_wav = label_wav[wav_start:wav_start + chunk_length]
        frame_num = (len(feat_wav) - win_size + win_size) // win_shift + 1
        frame_len = (frame_num - 1) * win_shift
        frame_mask_list.append(frame_num)

        feat_list.append(feat_wav)
        label_list.append(label_wav)
    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    #     if len(feat_wav) > chunk_length:
    #         wav_start = random.randint(0, len(feat_wav) - chunk_length)
    #         feat_wav = feat_wav[wav_start:wav_start + chunk_length]
    #         label_wav = label_wav[wav_start:wav_start + chunk_length]
    #
         # frame_num = (len(feat_wav) - win_size+win_size) // win_shift + 1
         # frame_len = (frame_num - 1) * win_shift
         # frame_mask_list.append(frame_num)
    #     feat_list.append(feat_wav)
    #     label_list.append(label_wav[:frame_len])
    #
    # feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    # label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    # feat_list = torch.stft(feat_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
    #                        window=torch.hann_window(fft_num)).permute(0,3,2,1)
    # label_list = torch.stft(label_list, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
    #                         window=torch.hann_window(fft_num)).permute(0,3,2,1)
    #
    # mag_feat, mag_label = torch.norm(feat_list, dim=1) ** 1.0, torch.norm(label_list, dim=1) ** 1.0
    # phase_feat, phase_label = torch.atan2(feat_list[:, -1, :, :], feat_list[:, 0, :, :]), torch.atan2(label_list[:, -1, :, :], label_list[:, 0, :, :])
    # feat_list, label_list = torch.stack((mag_feat * torch.cos(phase_feat), mag_feat * torch.sin(phase_feat)), dim=1), \
    #                           torch.stack((mag_label * torch.cos(phase_label), mag_label * torch.sin(phase_label)), dim=1)
    return feat_list, label_list, frame_mask_list


class CvDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=1,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = cv_generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader

class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list