import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np


# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


class dpcrn(nn.Module):
    def __init__(self):
        super(dpcrn, self).__init__()
        self.en = Encoder()
        self.dprnn = DPRNN()
        self.de = Decoder()

    def forward(self, inpt):
        x = inpt
        batch_size, _, seq_len, _ = x.shape
        x, x_list = self.en(x) #(B,C,T,F)

        x = self.dprnn(x)
        x = self.dprnn(x)

        mask = self.de(x, x_list)

        mask_real = mask[:, 0, :, :]
        mask_imag = mask[:, 1, :, :]
        inpt_real = inpt[:, 0, :, :]
        inpt_imag = inpt[:, 1, :, :]


        enh_real = inpt_real * mask_real - inpt_imag * mask_imag
        enh_imag = inpt_real * mask_imag + inpt_imag * mask_real

        return torch.stack((enh_real,enh_imag), dim=1)

class DPRNN(nn.Module):
    def __init__(self, in_features=None, out_features=None, mid_features=None, hidden_size=1024, groups=2):
        super(DPRNN, self).__init__()

#        hidden_size_t = hidden_size // groups

        self.intra_rnn = nn.LSTM(128, 64, 2, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(128,128)
        self.inter_rnn = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=False)
        self.inter_fc = nn.Linear(128,128)

        self.ln1 = nn.LayerNorm([4,128])
        self.ln2 = nn.LayerNorm([4,128])


    def forward(self, x):
        out = x #(B,C,T,F)
        ## intra
        # input shape (B,C,T,F) --> (B,T,F,C)
        x = x.permute(0, 2, 3, 1).contiguous()
        batch_size, chan_len, seq_len, freq_len = out.shape
        # input shape (B,C,T,F) --> (B*T,F,C)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(-1, freq_len, chan_len)
        # (bs*T,F,C)
        out, _ = self.intra_rnn(out)
        out = self.intra_fc(out)
        # (B*T,F,C) --> (B,T,F,C)
        out = out.view(batch_size, -1, freq_len, chan_len)
        out = self.ln1(out)
        intra_out = out + x #(B,T,F,C)

        ##inter
        # input shape (B,T,F, C) --> (B * F, T, C)
        out = intra_out.permute(0, 2, 1, 3).contiguous()
        out = out.view(-1, seq_len, chan_len)
        # (bs*F,T,C)
        out, _ = self.inter_rnn(out)
        out = self.inter_fc(out)
        # (B*F,T,C) --> (B,T,F,C)
        out = out.view(batch_size, -1, seq_len, chan_len)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.ln2(out)
        out = out + intra_out #(B,T,F,C)

        # output shape (B,T,F,C) --> (B,C,T,F)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pad1 = nn.ConstantPad2d((0, 0, 1, 0), value= 0.)
        en1 = nn.Sequential(
           pad1,
           nn.Conv2d(2, 32, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(32),
           nn.PReLU())
        en2 = nn.Sequential(
           pad1,
           nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(32),
           nn.PReLU())
        en3 = nn.Sequential(
           pad1,
           nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(32),
           nn.PReLU())
        en4 = nn.Sequential(
           pad1,
           nn.Conv2d(32, 64, kernel_size=(2, 3),stride=(1, 2)),
           nn.BatchNorm2d(64),
           nn.PReLU())
        en5 = nn.Sequential(
           pad1,
           nn.Conv2d(64, 128, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(128),
           nn.PReLU())
        self.en_module = nn.ModuleList([en1, en2, en3, en4, en5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en_module)):
            x = self.en_module[i](x)
            x_list.append(x)
        return x, x_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        de1 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 64, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(64),
            nn.PReLU())
        de2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.PReLU())
        de3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.PReLU())
        de4 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32, kernel_size=(2, 3), stride=(1, 2)),
            pad1,
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.PReLU())
        de5 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 2, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1))
        self.de_module = nn.ModuleList([de1, de2, de3, de4, de5])

    def forward(self, x, x_list):
        for i in range(len(self.de_module)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de_module[i](x)
        return x.squeeze(dim=1)


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t
    def forward(self, x):
        return x[:, :, 0:-self.chomp_t, :]