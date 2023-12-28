"""
This script is to enhance the audio data using the trained model
Date: 2019/06
Author: Andong Li
"""
import torch
import argparse
import librosa
import os
import numpy as np
import json
import scipy
import pickle
from data import *
from gaf_net_320 import gaf_base
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def enhance(args):
#    model = gaf_base(3, 64, 2, 4, 4, [1, 2, 5, 9], 64 * 7 + 257 * 2, 64 * 7, 64 * 7, (2, 3), (1, 3), 64, 'cat',
#                            2, is_aux=False, encoder_type='U2Net',
#                            tcm_type='full-band')
    model = gaf_base(3, 64, 2, 4, 4, [1, 2, 5, 9], 256 + 161 * 2, 256, 256, (2, 3), (1, 3), 64, 'cat', 3, is_aux=False, encoder_type='U2Net', tcm_type='full-band')
    model.load_state_dict(torch.load(args.Model_path))
#    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
#        clean_file_path = args.clean_file_path
        esti_file_path = args.esti_file_path
        os.makedirs(esti_file_path, exist_ok=True)
        file_list = os.listdir(mix_file_path)
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
#            feat_wav=feat_wav[:,0]
#            librosa.output.write_wav(os.path.join(esti_file_path, file_id), feat_wav, args.fs)
#            clean_file_id = file_id[6:]
#            clean_file_id = 'direct' + clean_file_id
#            clean_wav, _ = sf.read(os.path.join(clean_file_path, file_id))#clean_file_id
            c = np.sqrt(np.sum(feat_wav ** 2.0) / len(feat_wav))
            feat_wav = feat_wav / c
#            clean_wav = clean_wav/c

            feat_wav_1 = feat_wav

            feat_x_1 = librosa.stft(feat_wav_1, n_fft=fft_num, hop_length=win_shift, window='hanning').T
#            clean_x = librosa.stft(clean_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

            # use 1/3 compressed scale for magnitude
            feat_mag_x_1, feat_phase_x_1 = np.abs(feat_x_1) ** (1 / 2), np.angle(feat_x_1)
#            clean_mag_x_1, clean_phase_x_1 = np.abs(clean_x) ** (1 / 2), np.angle(clean_x)

#            feat_x_1 = feat_mag_x_1 * np.exp(1j * feat_phase_x_1)
#            feat_x = np.concatenate((np.real(feat_x_1)[:, :, np.newaxis].astype(np.float32),
#                               fa_de      np.imag(feat_x_1)[:, :, np.newaxis].astype(np.float32)), axis=-1)
#            feat_x = torch.FloatTensor(feat_x)
#            feat_x = feat_x.permute(2, 0, 1).contiguous()
            feat_x = torch.stack((torch.FloatTensor(feat_mag_x_1 * np.cos(feat_phase_x_1)), torch.FloatTensor(feat_mag_x_1 * np.sin(feat_phase_x_1))), dim=0)
#            clean_phase_x_1 = torch.FloatTensor(clean_phase_x_1).cuda()
            feat_x = feat_x.cuda()
#            clean_x = torch.stack((torch.FloatTensor(clean_mag_x_1 * np.cos(clean_phase_x_1)), torch.FloatTensor(clean_mag_x_1 * np.sin(clean_phase_x_1))), dim=0)
#            clean_x = clean_x.cuda()
            esti_x_list = model(feat_x.unsqueeze(dim=0).cuda())
##            esti_x_list_mag, esti_x_list = model(feat_x.unsqueeze(dim=0).cuda())
###            esti_x = esti_x_list_RI[-2].squeeze(dim=0)
            esti_x = esti_x_list[-1].squeeze(dim=0)
#            esti_x_mag = esti_x_mag.permute(1, 0)
#            esti_x_mag = esti_x_mag**2
##            esti_mag_x = esti_x_mag_list[-3].squeeze(dim=0)
            esti_x = esti_x.squeeze(dim=0).permute(0, 2, 1)
##            esti_mag_x = esti_mag_x.permute(1, 0)
            # use 1/3 compressed scale for magnitude
            esti_mag_x, esti_phase_x = torch.norm(esti_x, dim=0) ** (2 / 1), torch.atan2(esti_x[1, :, :], esti_x[0, :, :])
            ####            esti_mag_x, esti_phase_x = esti_mag_x ** (2 / 1), torch.atan2(esti_x[1, :, :], esti_x[0, :, :])
#            clean_mag_x = torch.norm(clean_x, dim=0) ** (2 / 1)
#            clean_phase_x = torch.atan2(clean_x[1, :, :], clean_x[0, :, :])
#            reverb_phase_x = torch.atan2(feat_x[1, :, :], feat_x[0, :, :])
            esti_real_x, esti_imag_x = esti_mag_x * torch.cos(esti_phase_x), esti_mag_x * torch.sin(esti_phase_x)#esti_phase_x
            esti_real_x, esti_imag_x = esti_real_x.cpu().numpy(), esti_imag_x.cpu().numpy()
            de_esti = esti_real_x + 1j * esti_imag_x

            #de_esti = esti_x[0, :, :] + 1j * esti_x[1, :, :]
            esti_utt = librosa.istft((de_esti).T, hop_length=win_shift,
                                     win_length=win_size, window='hanning', length=len(feat_wav))
            esti_utt = esti_utt * c
#            librosa.output.write_wav(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)

            print(' The %d utterance has been decoded!' % (cnt+1))
            cnt = cnt + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
#    parser.add_argument('--mix_file_path', type=str, default='/media/luoxiaoxue/LXX/dereverb_rir/single/datasets/test/new_test_datasets/Reverb_Challenge/LargeRoom/reverb')#real/near/reverb real_shiyanshi/RoomB/reverb
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/datasets/VB_DEMAND_48K/noisy_16k')
    parser.add_argument('--esti_file_path', type=str,
                        default='./datasets/vb_gaf_cprs_model')
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str,  default='./BEST_MODEL/vb_gaf_cprs_model.pth',#complex_compressed_1_2_dereverb_model_dir_image_new
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)
