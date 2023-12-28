import torch
import argparse
import os
import numpy as np
#from istft import ISTFT
from Step1_network import Step1_net
from Step2_network import Step2_net

import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def enhance(args):
    model1 = Step1_net()
    model2 = Step2_net(X=6, R=3)
    model1.load_state_dict(torch.load('./BEST_MODEL/step1_vb_cts_noncprs_model_final.pth'))
    model2.load_state_dict(torch.load('./BEST_MODEL/step2_vb_cts_noncprs_model.pth'))
    model1.cuda().eval()
    model2.cuda().eval()


    # with torch.no_grad():
    #     cnt = 0
    #     mix_file_path = args.mix_file_path
    #     esti_file_path = args.esti_file_path
    #     os.makedirs(esti_file_path, exist_ok=True)
    #     file_list = os.listdir(mix_file_path)
    #     istft = ISTFT(filter_length=320, hop_length=160, window='hanning')
    #     for file_id in file_list:
    #         feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
    #         c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
    #         feat_wav = feat_wav * c
    #         wav_len = len(feat_wav)
    #         frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
    #         fake_wav_len = (frame_num - 1) * 160 + 320 - 320
    #         left_sample = fake_wav_len - wav_len
    #         feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
    #         feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
    #                             window=torch.hann_window(320)).permute(0, 3, 2, 1)
    #         feat_x, phase_x = feat_x.cuda(), torch.atan2(feat_x[:,1,:,:], feat_x[:,0,:,:]).cuda()
    #
    #         # the first step
    #         esti_x = model1(torch.norm(feat_x, dim=1))
    #         s1_esti_real, s1_esti_imag = esti_x * torch.cos(phase_x), esti_x * torch.sin(phase_x)
    #         s1_com_out = torch.stack((s1_esti_real, s1_esti_imag), dim=1)
    #         s2_in = torch.cat((feat_x, s1_com_out), dim=1)
    #         s2_esti_out = model2(s2_in)
    #         s2_esti_out = s2_esti_out + s1_com_out
    #         esti_utt = istft(s2_esti_out).squeeze().cpu().numpy()
    #         esti_utt = esti_utt[:wav_len]
    #         esti_utt = esti_utt / c
    #         sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
    #         print(' The %d utterance has been decoded!' % (cnt + 1))
    #         cnt += 1

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path
        file_list = os.listdir(mix_file_path)

        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x_ = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1)
            # compressed
            feat_x, phase_x = torch.norm(feat_x_, dim=1) ** 1.0, torch.atan2(feat_x_[:, 1, :, :], feat_x_[:, 0, :, :])

            feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=1)
            feat_x, phase_x = feat_x.cuda(), phase_x.cuda()

            # the first step
            esti_x = model1(torch.norm(feat_x, dim=1))
            s1_esti_real, s1_esti_imag = esti_x * torch.cos(phase_x), esti_x * torch.sin(phase_x)
            s1_com_out = torch.stack((s1_esti_real, s1_esti_imag), dim=1)
            s2_in = torch.cat((feat_x, s1_com_out), dim=1)
            s2_esti_out = model2(s2_in)
            s2_esti_out = s2_esti_out + s1_com_out

            # compressed
            esti_x_mag = torch.norm(s2_esti_out, dim=1) ** 1.

            esti_x_phase = torch.atan2(s2_esti_out[:, 1, :, :], s2_esti_out[:, 0, :, :])
            s2_esti_out = torch.stack((esti_x_mag * torch.cos(esti_x_phase), esti_x_mag * torch.sin(esti_x_phase)), dim=1)

            s2_esti_out = s2_esti_out.permute(0,3,2,1).cpu()
            esti_utt = torch.istft(s2_esti_out, 320, 160, 320, torch.hann_window(320))
            esti_utt = esti_utt.squeeze(dim=0)
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(esti_file_path, exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/media/luoxiaoxue/datasets/VB_DEMAND_48K/noisy_16k')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/vb_cts_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    args = parser.parse_args()
    enhance(args=args)
