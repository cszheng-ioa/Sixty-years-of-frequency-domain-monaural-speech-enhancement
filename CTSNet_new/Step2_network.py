import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from Backup import numParams
#from ptflops import get_model_complexity_info
#from thop import profile


class Step2_net(nn.Module):
    def __init__(self, X, R):
        super(Step2_net, self).__init__()
        self.X = X
        self.R = R
        self.en = Encoder()
        self.de_r = Decoder()
        self.de_i = Decoder()
        self.tcm_list = nn.ModuleList([Glu_list(X=self.X) for r in range(self.R)])

    def forward(self, inpt):
        x, x_list = self.en(inpt)
        batch_num, _, seq_len, _ = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_num, -1, seq_len)
        x_acc = Variable(torch.zeros(x.size()), requires_grad=True).to(x.device)
        for i in range(len(self.tcm_list)):
            x = self.tcm_list[i](x)
            x_acc = x_acc + x
        x = x_acc
        x = x.view(batch_num, 64, 4, seq_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x_r = self.de_r(x, x_list)
        x_i = self.de_i(x, x_list)
        del x_list
        return torch.stack((x_r, x_i), dim=1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        en1 = nn.Sequential(
            Gate_Conv(4, 64, kernel_size=(2, 5), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        en2 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        en3 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        en4 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        en5 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
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
        de1 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1, pad=(1, 0, 0, 0)),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        de2 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        de3 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        de4 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            CumulativeLayerNorm2d(num_features=64, affine=True),
            nn.PReLU(64)
        )
        de5 = nn.Sequential(
            Gate_Conv(64*2, 1, kernel_size=(2, 5), stride=(1, 2), de_flag=1, chomp=1),
            CumulativeLayerNorm2d(num_features=1, affine=True),
            nn.PReLU(1)
        )
        self.de_list = nn.ModuleList([de1, de2, de3, de4, de5])
        self.de6 = nn.Sequential(
            nn.Linear(161, 161))

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de_list[i](x)
        x = self.de6(x.squeeze(dim=1))
        return x

class Glu_list(nn.Module):
    def __init__(self, X):
        super(Glu_list, self).__init__()
        self.X = X
        self.glu_list = nn.ModuleList([glu(i) for i in range(self.X)])
    def forward(self, x):
        for i in range(self.X):
            x = self.glu_list[i](x)
        return x

class glu(nn.Module):
    def __init__(self, dila_rate):
        super(glu, self).__init__()
        ori_dila = 2**dila_rate
        self.in_conv = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.ori_conv = nn.Sequential(
            nn.PReLU(64),
            CumulativeLayerNorm1d(num_features=64, affine=True),
            ShareSepConv(2*ori_dila - 1),
            nn.ConstantPad1d((4*ori_dila, 0), value=0.),
            nn.Conv1d(64, 64, kernel_size=5, dilation=ori_dila, bias=False)
        )
        self.att_ori = nn.Sequential(
            nn.PReLU(64),
            CumulativeLayerNorm1d(num_features=64, affine=True),
            ShareSepConv(2*ori_dila - 1),
            nn.ConstantPad1d((4*ori_dila, 0), value=0.),
            nn.Conv1d(64, 64, kernel_size=5, dilation=ori_dila, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(64),
            CumulativeLayerNorm1d(num_features=64, affine=True),
            nn.Conv1d(64, 256, kernel_size=1, bias=False)
        )

    def forward(self, x):
        inpt = x
        x = self.in_conv(x)
        x = self.ori_conv(x) * self.att_ori(x)
        x = self.out_conv(x)
        x = x + inpt
        return x



class Gate_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag, pad=(0, 0, 0, 0), chomp=1):
        super(Gate_Conv, self).__init__()
        if de_flag == 0:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride))
            self.gate_conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp))
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels,
                                                kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp),
                nn.Sigmoid())
    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        self.pad = nn.ConstantPad1d((kernel_size-1, 0), value=0.)
        weight_tensor = torch.zeros(1, 1, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size).contiguous()
        x = self.pad(x)
        x = F.conv1d(x, expand_weight, None, stride=1, dilation=1, groups=inc)
        return x

class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t
    def forward(self, x):
        return x[:, :, 0:-self.t, :]

class CumulativeLayerNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        else:
            self.gain = Variable(torch.ones(1,num_features,1,1), requires_grad=False)
            self.bias = Variable(torch.zeros(1,num_features,1,1), requires_grad=False)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1,3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1,3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = np.arange(channel*freq_num, channel*freq_num*(seq_len+1), channel*freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

class CumulativeLayerNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1,num_features,1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,num_features,1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = np.arange(channel, channel*(seq_len+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(inpt)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


if __name__ == '__main__':
    net_work = Step2_net(X=6, R=3)
    net_work.cuda()
    net_work.eval()
    print('The number of trainable parameters of the net is:%d' % (numParams(net_work)))
    input = torch.randn(1, 4, 1, 161).cuda()