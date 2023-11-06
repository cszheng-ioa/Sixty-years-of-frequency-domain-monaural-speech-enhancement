#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

EPSILON = torch.finfo(torch.float32).eps

from f_att_cplx import Multihead_Attention_F_Branch
from t_att_cplx import Multihead_Attention_T_Branch
from f_att_real import Multihead_Attention_F_Branch_real
from t_att_real import Multihead_Attention_T_Branch_real
from dsconv2d_cplx import DSConv2d
from dsconv2d_real import DSConv2d_Real
from ff_real import FF_Real
from ff_cplx import FF_Cplx
from fusion import fusion as fusion
from show import show_model, show_params


class Dilated_Dualpath_Conformer(nn.Module):

    def __init__(self, inchannel=128, hiddenchannel=64):
        super(Dilated_Dualpath_Conformer, self).__init__()
        
        self.ff1_cplx = FF_Cplx(inchannel, hiddenchannel)
        self.ff1_mag = FF_Real(inchannel, hiddenchannel)
        
        
        self.cplx_tatt = Multihead_Attention_T_Branch(inchannel, 16)
        self.cplx_fatt = Multihead_Attention_F_Branch(inchannel, 16)
        self.mag_tatt = Multihead_Attention_T_Branch_real(inchannel, 16)
        self.mag_fatt = Multihead_Attention_F_Branch_real(inchannel, 16)

    
        dilation = [1, 2, 4, 8, 16, 32, 64, 128]
        self.dsconv_cplx = nn.ModuleList()
        for idx in range(len(dilation)):
            self.dsconv_cplx.append(DSConv2d(inchannel, 32, dilation1=dilation[idx], dilation2=dilation[len(dilation)-idx-1]))
        self.dsconv_real = nn.ModuleList()
        for idx in range(len(dilation)):
            self.dsconv_real.append(DSConv2d_Real(inchannel, 32, dilation1=dilation[idx], dilation2=dilation[len(dilation)-idx-1]))


        self.ff2_cplx = FF_Cplx(inchannel, hiddenchannel)
        self.ff2_mag = FF_Real(inchannel, hiddenchannel)

        self.ln_conformer_cplx = nn.LayerNorm(inchannel)
        self.ln_conformer_mag = nn.LayerNorm(inchannel)

    def forward(self, cplx, mag):
        # N C F T 2
        # N C F T

        cplx = self.ff1_cplx(cplx)
        mag= self.ff1_mag(mag)
        cplx, mag = fusion(cplx, mag)

        cplx = self.cplx_tatt(cplx)
        mag = self.mag_tatt(mag)
        cplx, mag = fusion(cplx, mag)

        cplx = self.cplx_fatt(cplx)
        mag = self.mag_fatt(mag)
        cplx, mag = fusion(cplx, mag)

        for idx in range(len(self.dsconv_cplx)):
            cplx = self.dsconv_cplx[idx](cplx)
            mag = self.dsconv_real[idx](mag)
            cplx, mag = fusion(cplx, mag)

        cplx = self.ff2_cplx(cplx)
        mag= self.ff2_mag(mag)
        cplx, mag = fusion(cplx, mag)
        cplx, mag = self.ln_conformer_cplx(cplx.transpose(1,4)).transpose(1,4), self.ln_conformer_mag(mag.transpose(1,3)).transpose(1,3)
        return cplx, mag


if __name__ == '__main__':
    net = Dilated_Dualpath_Conformer(128, 64)
    inputs = torch.ones([10, 128, 4, 397, 2])
    y = net(inputs, inputs[...,0])
    print(y[0].shape, y[1].shape)
