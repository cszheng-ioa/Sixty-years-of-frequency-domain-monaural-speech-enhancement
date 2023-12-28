import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Step2_config import win_size, win_shift, fft_num
from Backup import numParams
import torch.nn.functional as F
#from ptflops import get_model_complexity_info

class gaf_base(nn.Module):
    def __init__(self,
                 kd1,
                 cd1,
                 tcm_num,
                 sub_g1,
                 sub_g2,
                 dilas,
                 ci,
                 co1,
                 co2,
                 k1,
                 k2,
                 c,
                 intra_connect,
                 stage_num=3,
                 is_causal=True,
                 is_aux=True,
                 encoder_type='U2Net',
                 tcm_type='full-band'):
        """
        This class is the basic module of glance-and-focus network, which follows the 'glance and focus' strategy for\
        speech enhancement task. It consists of several stages, each of which includes two separate branches, namely
        glance branch and focus branch.
        :param kd1: the kernel size of the MS-TCM, default: 3
        :param cd1: the channel size of the MS-TCM, default: 64
        :param tcm_num: the number of MS-TCM in each branch
        :param sub_g1: the number of groups in the glance branch, default: 4  (256 // 64)
        :param sub_g2: the number of groups in the focus branch, default: 4 (256//64)
        :param dilas: the dilation list, default: [1, 2, 5, 9]
        :param ci:  the channel size of the input feature size, default: 256 + (161*2)
        :param co1:  the channel size of the intermediate feature size in the glance branch, default: 256
        :param co2:  the channel size of the intermediate feature size in the focus branch, default: 256
        :param k1:  the kernel size of the 2-D inter-convolutional layers within the encoder, default: (2, 3)
        :param k2:  the kernel size of the 2-D intra-convolutional layers within the encoder, default: (1, 3)
        :param c:  the channel size of the 2-D convolutional layers, default: 64
        :param intra_connect: the connection method of the intra-connection, default: 'cat'
        :param stage_num:  the number of stages of the GaF module, default: 3
        :param is_causal: whether the system is implemented with causal mechanism, default: True
        :param is_aux: whether the auxilary decoder module with deconvolutional layers are applied, default: True
        :param encoder_type: the type of the encoder, choices: UNet and U2Net
        :param tcm_type: the type of the tcm, choices: full-band and sub-band
        """

        super(gaf_base, self).__init__()
        self.kd1, self.cd1, self.tcm_num, self.sub_g1, self.sub_g2 = kd1, cd1, tcm_num, sub_g1, sub_g2
        self.dilas, self.ci, self.co1, self.co2 = dilas, ci, co1, co2
        self.k1, self.k2, self.c, self.intra_connect = k1, k2, c, intra_connect
        self.stage_num = stage_num
        self.is_causal = is_causal
        self.is_aux = is_aux
        self.encoder_type = encoder_type
        self.tcm_type = tcm_type
        if self.encoder_type == 'U2Net':
            self.en = U2Net_Encoder(self.k1, self.k2, self.c, self.intra_connect)
        elif self.encoder_type == 'UNet':
            self.en = UNet_Encoder(self.k1, self.c)

        self.gafs = nn.ModuleList([GAF_module(self.kd1, self.cd1, self.tcm_num, self.sub_g1, self.sub_g2, self.dilas,
                                              self.ci, self.co1, self.co2, self.is_causal, self.tcm_type) for _ in range(self.stage_num)])
        if self.is_aux:
            self.aux_de = Aux_decoder(self.k1, self.c)

    def forward(self, inpt):
        batch_num, _, seq_len, _ = inpt.size()
        feat_x, x_list = self.en(inpt)
        x = feat_x.transpose(-2, -1).contiguous()
        x = x.view(batch_num, -1, seq_len)
        pre_x = inpt.transpose(-2, -1).contiguous()
        out_list = []
        for i in range(len(self.gafs)):
            tmp = self.gafs[i](x, pre_x)
            pre_x = tmp
            if i == len(self.gafs)-1 and self.is_aux:
                x_aux = self.aux_de(feat_x, x_list)
                tmp = tmp + x_aux
            out_list.append(tmp)
        return out_list


class GAF_module(nn.Module):
    def __init__(self, kd1, cd1, tcm_num, sub_g1, sub_g2, dilas, ci, co1, co2, is_causal, tcm_type):
        super(GAF_module, self).__init__()
        self.kd1, self.cd1, self.tcm_num, self.sub_g1, self.sub_g2 = kd1, cd1, tcm_num, sub_g1, sub_g2
        self.dilas = dilas
        self.ci, self.co1, self.co2 = ci, co1, co2
        self.is_causal = is_causal
        self.tcm_type = tcm_type
        self.glance_branch = Glance_branch(self.kd1, self.cd1, self.tcm_num, sub_g1, self.dilas, self.ci,
                                           self.co1, self.is_causal, self.tcm_type)
        self.focus_branch = Focus_branch(self.kd1, self.cd1, self.tcm_num, sub_g2, self.dilas, self.ci,
                                         self.co2, self.is_causal, self.tcm_type)


    def forward(self, feat_x, pre_x):
        # pre_x: (B, 2, C, T)
        batch_num, _, c, seq_len = pre_x.size()
        pre_mag, pre_phase = torch.norm(pre_x, dim=1), torch.atan2(pre_x[:,-1,:,:], pre_x[:,0,:,:])
        flatten_pre = pre_x.view(batch_num, -1, seq_len)
        x = torch.cat((feat_x, flatten_pre), 1)
        gain_filter = self.glance_branch(x)   # (B, L, C, T)
        com_resi = self.focus_branch(x)  # (B, 2, C, T)
        x_mag = pre_mag * gain_filter
        x_r, x_i = x_mag*torch.cos(pre_phase), x_mag*torch.sin(pre_phase)
        x = torch.stack((x_r, x_i), 1) + com_resi
        return x


class Glance_branch(nn.Module):
    def __init__(self, kd1, cd1, tcm_num, sub_group, dilas, ci, co, is_causal, tcm_type):
        super(Glance_branch, self).__init__()
        self.kd1, self.cd1, self.tcm_num, self.sub_group, self.dilas = kd1, cd1, tcm_num, sub_group, dilas
        self.ci, self.co = ci, co
        self.is_causal = is_causal
        self.tcm_type = tcm_type
        self.in_conv_main = nn.Conv1d(self.ci, self.co, 1)
        self.in_conv_gate = nn.Sequential(
            nn.Conv1d(self.ci, self.co, 1),
            nn.Sigmoid()
        )
        tcm_list = []
        for _ in range(self.tcm_num):
            if self.tcm_type == 'sub-band':
                tcm_list.append(Ms_TCM(self.kd1, self.cd1, self.sub_group, self.dilas, self.is_causal))
            elif self.tcm_type == 'full-band':
                tcm_list.append(Tcm_list(self.dilas, self.is_causal))
        self.mstcm_filter = nn.Sequential(
            *tcm_list,
            nn.Conv1d(self.co, 161, 1),
            nn.Sigmoid()
        )

    def forward(self, inpt):
        batch_num, _, seq_len = inpt.size()
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        x = self.mstcm_filter(x)
        return x


class Focus_branch(nn.Module):
    def __init__(self, kd1, cd1, tcm_num, sub_group, dilas, ci, co, is_causal, tcm_type):
        super(Focus_branch, self).__init__()
        self.kd1, self.cd1, self.tcm_num, self.sub_group, self.dilas = kd1, cd1, tcm_num, sub_group, dilas
        self.ci, self.co = ci, co
        self.is_causal = is_causal
        self.tcm_type = tcm_type
        self.in_conv_main = nn.Conv1d(self.ci, self.co, 1)
        self.in_conv_gate = nn.Sequential(
            nn.Conv1d(self.ci, self.co, 1),
            nn.Sigmoid()
        )
        tcm_list_r, tcm_list_i = [], []
        for _ in range(self.tcm_num):
            if self.tcm_type == 'sub-band':
                tcm_list_r.append(Ms_TCM(self.kd1, self.cd1, self.sub_group, self.dilas, self.is_causal))
                tcm_list_i.append(Ms_TCM(self.kd1, self.cd1, self.sub_group, self.dilas, self.is_causal))
            elif self.tcm_type == 'full-band':
                tcm_list_r.append(Tcm_list(self.dilas, self.is_causal))
                tcm_list_i.append(Tcm_list(self.dilas, self.is_causal))

        self.mstcm_r = nn.Sequential(
            *tcm_list_r,
            nn.Conv1d(self.co, 161, 1)
        )
        self.mstcm_i = nn.Sequential(
            *tcm_list_i,
            nn.Conv1d(self.co, 161, 1)
        )

    def forward(self, inpt):
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        x_r, x_i = self.mstcm_r(x), self.mstcm_i(x)
        x = torch.stack((x_r, x_i), dim=1)
        return x


class Ms_TCM(nn.Module):
    def __init__(self, kd1, cd1, sub_group, dilas, is_causal=True):
        super(Ms_TCM, self).__init__()
        self.kd1, self.cd1, self.sub_group, self.dilas = kd1, cd1, sub_group, dilas
        self.is_causal = is_causal
        en_unit_list, de_unit_list = [], []
        for i in range(self.sub_group):
            if i == 0:
                en_unit_list.append(Conv1dunit(self.cd1, self.cd1, self.kd1, self.dilas[i], self.is_causal))
                de_unit_list.append(Conv1dunit(self.cd1, self.cd1, self.kd1, self.dilas[i], self.is_causal))
            else:
                en_unit_list.append(Conv1dunit(2*self.cd1, self.cd1, self.kd1, self.dilas[np.mod(i, len(self.dilas))],
                                               self.is_causal))
                de_unit_list.append(Conv1dunit(2*self.cd1, self.cd1, self.kd1, self.dilas[np.mod(i, len(self.dilas))],
                                               self.is_causal))
        self.en_unit_list, self.de_unit_list = nn.ModuleList(en_unit_list), nn.ModuleList(de_unit_list)


    def forward(self, inpt):
        # split the tensor into several sub-bands
        batch_size, channel_num, seq_len = inpt.size()
        inpt = inpt.view(batch_size, self.sub_group, -1, seq_len)
        forward_out, backward_out = Variable(torch.zeros_like(inpt)).to(inpt.device), \
                                    Variable(torch.zeros_like(inpt)).to(inpt.device)

        # begin the forward encoding part
        for i in range(len(self.en_unit_list)):
            if i == 0:
                x = self.en_unit_list[i](inpt[:,i,:,:])
            else:
                x = torch.cat((x, inpt[:,i,:,:]), dim=1)
                x = self.en_unit_list[i](x)
            forward_out[:,i,:,:] = x

        # begin the backward decoding part
        for i in range(len(self.de_unit_list)):
            if i == 0:
                x = self.de_unit_list[i](x)
            else:
                x = torch.cat((x, inpt[:,-(i+1),:,:]), dim=1)
                x = self.de_unit_list[i](x)
            backward_out[:,-(i+1),:,:] = x

        x = forward_out + backward_out
        x = x.view(batch_size, -1, seq_len)
        return x

class Tcm_list(nn.Module):
    def __init__(self, dilas, is_causal=True):
        super(Tcm_list, self).__init__()
        self.dilas, self.is_causal = dilas, is_causal
        self.tcm_list = nn.ModuleList([Glu(dilas[i], self.is_causal) for i in range(len(dilas))])

    def forward(self, x):
        for i in range(len(self.tcm_list)):
            x = self.tcm_list[i](x)
        return x


class Glu(nn.Module):
    def __init__(self, dilation, is_causal=True):
        super(Glu, self).__init__()
        self.in_conv = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        if is_causal:
            self.left_conv = nn.Sequential(
                nn.PReLU(64),
                nn.InstanceNorm1d(64, affine=True),
                nn.ConstantPad1d((2*dilation, 0), value=0.),
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False)
                )
        else:
            self.left_conv = nn.Sequential(
                nn.PReLU(64),
                nn.InstanceNorm1d(64, affine=True),
                nn.ConstantPad1d((dilation, dilation), value=0.),
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation, bias=False)
                )
        self.out_conv = nn.Sequential(
            nn.PReLU(64),
            nn.InstanceNorm1d(64, affine=True),
            nn.Conv1d(64, 256, kernel_size=1, bias=False)
            )
    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.left_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class U2Net_Encoder(nn.Module):
    def __init__(self, k1, k2, c, intra_connect):
        super(U2Net_Encoder, self).__init__()
        self.k1, self.k2, self.c, self.intra_connect = k1, k2, c, intra_connect
        meta_unet = []
        meta_unet.append(
            En_unet_module((2, 5), self.k2, self.c, self.intra_connect, scale=4, de_flag=False, is_first=True))
        meta_unet.append(
            En_unet_module(self.k1, self.k2, self.c, self.intra_connect, scale=3, de_flag=False))
        meta_unet.append(
            En_unet_module(self.k1, self.k2, self.c, self.intra_connect, scale=2, de_flag=False))
        meta_unet.append(
            En_unet_module(self.k1, self.k2, self.c, self.intra_connect, scale=1, de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            Gate_2dconv(self.c, 64, self.k1, stride=(1, 2), de_flag=False, pad=(0,0,1,0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )

    def forward(self, x):
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list

class UNet_Encoder(nn.Module):
    def __init__(self, k1, c):
        super(UNet_Encoder, self).__init__()
        self.k1, self.c = k1, c
        unet = []
        unet.append(nn.Sequential(
            Gate_2dconv(2, self.c, (2, 5), (1, 2), de_flag=False, pad=(0,0,1,0)),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)))
        unet.append(nn.Sequential(
            Gate_2dconv(self.c, self.c, self.k1, (1, 2), de_flag=False, pad=(0,0,1,0)),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)))
        unet.append(nn.Sequential(
            Gate_2dconv(self.c, self.c, self.k1, (1, 2), de_flag=False, pad=(0,0,1,0)),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)))
        unet.append(nn.Sequential(
            Gate_2dconv(self.c, self.c, self.k1, (1, 2), de_flag=False, pad=(0,0,1,0)),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)))
        unet.append(nn.Sequential(
            Gate_2dconv(self.c, 64, self.k1, (1, 2), de_flag=False, pad=(0,0,1,0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x):
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class Aux_decoder(nn.Module):
    def __init__(self, k1, c):
        super(Aux_decoder, self).__init__()
        self.k1, self.c = k1, c
        de1 = nn.Sequential(
            Gate_2dconv(64, self.c, self.k1, (1, 2), de_flag=True),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)
        )
        de2 = nn.Sequential(
            Gate_2dconv(self.c, self.c, self.k1, (1, 2), de_flag=True),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)
        )
        de3 = nn.Sequential(
            Gate_2dconv(self.c, self.c, self.k1, (1, 2), de_flag=True),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)
        )
        de4 = nn.Sequential(
            Gate_2dconv(self.c, self.c, self.k1, (1, 2), de_flag=True),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)
        )
        de5 = nn.Sequential(
            Gate_2dconv(self.c, self.c, (2, 5), (1, 2), de_flag=True),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)
        )
        self.de_list = nn.ModuleList([de1, de2, de3, de4, de5])
        self.de6 = nn.Conv2d(self.c, 2, (1, 1))

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            if i == 0:
                x = self.de_list[i](x)
            else:
                tmp = x + x_list[-(i+1)]
                x = self.de_list[i](tmp)
        x = self.de6(x)
        return x.transpose(-2, -1).contiguous()


class En_unet_module(nn.Module):
    def __init__(self, k1, k2, c, intra_connect, scale, de_flag=False, is_first=False):
        super(En_unet_module, self).__init__()
        self.k1, self.k2 = k1, k2
        self.c = c
        self.intra_connect = intra_connect
        self.scale = scale
        self.de_flag = de_flag
        self.is_first = is_first
        in_conv_list = []

        if self.is_first:
            in_conv_list.append(Gate_2dconv(2, self.c, self.k1, (1, 2), self.de_flag, pad=(0,0,1,0)))
        else:
            in_conv_list.append(Gate_2dconv(self.c, self.c, self.k1, (1, 2), self.de_flag, pad=(0,0,1,0)))
        in_conv_list.append(nn.InstanceNorm2d(self.c, affine=True))
        in_conv_list.append(nn.PReLU(self.c))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(self.scale):
            enco_list.append(Conv2dunit(self.k2, self.c))
        for i in range(self.scale):
            if i == 0:
                deco_list.append(Deconv2dunit(self.k2, self.c, 'add'))
            else:
                deco_list.append(Deconv2dunit(self.k2, self.c, self.intra_connect))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(self.intra_connect)

    def forward(self, x):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi

class Conv2dunit(nn.Module):
    def __init__(self, k, c):
        super(Conv2dunit, self).__init__()
        self.k, self.c = k, c
        self.conv = nn.Sequential(
            nn.Conv2d(self.c, self.c, self.k, (1, 2)),
            nn.InstanceNorm2d(self.c, affine=True),
            nn.PReLU(self.c)
        )

    def forward(self, x):
        return self.conv(x)


class Deconv2dunit(nn.Module):
    def __init__(self, k, c, intra_connect):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        deconv_list = []
        if self.intra_connect == 'add':
            deconv_list.append(nn.ConvTranspose2d(self.c, self.c, self.k, (1, 2)))
        elif self.intra_connect == 'cat':
            deconv_list.append(nn.ConvTranspose2d(2*self.c, self.c, self.k, (1, 2)))
        deconv_list.append(nn.InstanceNorm2d(self.c, affine=True))
        deconv_list.append(nn.PReLU(self.c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, x):
        return self.deconv(x)


class Gate_2dconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag, pad=(0,0,0,0), chomp=1):
        super(Gate_2dconv, self).__init__()
        if de_flag is False:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            self.gate_conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp))
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp),
                nn.Sigmoid())

    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)

class Conv1dunit(nn.Module):
    def __init__(self, ci, co, k, dila, is_causal):
        super(Conv1dunit, self).__init__()
        self.ci, self.co, self.k, self.dila = ci, co, k, dila
        self.is_causal = is_causal
        if self.is_causal:
            pad = nn.ConstantPad1d(((self.k-1)*self.dila, 0), value=0.)
        else:
            pad = nn.ConstantPad1d(((self.k-1)*self.dila//2, (self.k-1)*self.dila//2), value=0.)

        self.unit = nn.Sequential(
            pad,
            nn.Conv1d(self.ci, self.co, self.k, dilation=self.dila),
            nn.InstanceNorm1d(self.co, affine=True),
            nn.PReLU(self.co)
        )

    def forward(self, x):
        x = self.unit(x)
        return x

class Skip_connect(nn.Module):
    def __init__(self, connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == 'add':
            x = x_main + x_aux
        elif self.connect == 'cat':
            x = torch.cat((x_main, x_aux), dim=1)
        return x

class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t
    def forward(self, x):
        return x[:, :, 0:-self.t, :]


if __name__ == '__main__':
    model = gaf_base(3,64,2,4,4,[1,2,5,9],256+161*2,256,256,(2,3),(1,3),64,'cat',4, is_aux=False, encoder_type='U2Net',
                     tcm_type='full-band')
    #model = Focus_branch(kd1=3, cd1=64, tcm_num=2, sub_group=4, dilas=[1,2,5,9], ci=256+161*2, co=256, is_causal=True)
    #model = Ms_TCM(kd1=3, cd1=64, sub_group=8, dilas=[1,2,5,9], is_causal=True)
    model.eval()
    model.cuda()
    x = torch.FloatTensor(4, 2, 10, 161).cuda()
    x = model(x)
    print('The number of parameters of the model is:%.5d' % numParams(model))
    macs, params = get_model_complexity_info(model, (2, 100, 161), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)