import torch
import argparse
import os
import numpy as np
from TaylorSENet import TaylorSENet
import soundfile as sf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def enhance(args):
    model = TaylorSENet(cin=2, k1=(1,3), k2=(2,3), c=64, kd1=5, cd1=64, d_feat=256, dilations=[1,2,5,9], p=2, fft_num=320,
                        order_num=3, intra_connect='cat',inter_connect='cat',is_causal=True,is_conformer=False, is_u2=True,
                        is_param_share=False, is_encoder_share=False).cuda()
    checkpoint = torch.load('./BEST_MODEL/vb_taylor_cprs_model.pth')
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path

        file_list = os.listdir(mix_file_path)
        for file_id in file_list:
            file_name = file_id.split('.')[0]
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320)).permute(0, 3, 2, 1).cuda()

            #compressed
            feat_mag, feat_phase = torch.norm(feat_x, dim=1)**0.5, torch.atan2(feat_x[:,-1,:,:], feat_x[:,0,:,:])
            feat_x_compress = torch.stack((feat_mag*torch.cos(feat_phase), feat_mag*torch.sin(feat_phase)), dim=1)
            esti_x = model(feat_x_compress)
            #uncompressed
            esti_mag, esti_phase = torch.norm(esti_x, dim=1)**2.0, torch.atan2(esti_x[:,-1,:,:], esti_x[:,0,:,:])
            esti_x = torch.stack((esti_mag*torch.cos(esti_phase), esti_mag*torch.sin(esti_phase)), dim=1)
            esti_x = esti_x.permute(0,3,2,1)

            esti_utt = torch.istft(esti_x, 320, 160, 320, window=torch.hann_window(320).cuda(), length=wav_len).transpose(-2,-1).squeeze()
            esti_utt = esti_utt.cpu().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c

            os.makedirs(esti_file_path, exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)

            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

        # cnt = 0
        # mix_file_path = args.mix_file_path
        # esti_file_path = args.esti_file_path
        # file_list = os.listdir(mix_file_path)
        # for file_id in file_list:
        #     file_name = file_id.split('.')[0]
        #     feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
        #     c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
        #     feat_wav = feat_wav * c
        #     wav_len = len(feat_wav)
        #     frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
        #     fake_wav_len = (frame_num - 1) * 160 + 320 - 320
        #     left_sample = fake_wav_len - wav_len
        #     feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
        #     feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
        #                         window=torch.hann_window(320)).permute(0, 3, 2, 1).cuda()
        #
        #     feat_mag, feat_phase = torch.norm(feat_x, dim=1) ** 0.5, torch.atan2(feat_x[:, -1, :, :],
        #                                                                             feat_x[:, 0, :, :])
        #     feat_x_compress = torch.stack((feat_mag * torch.cos(feat_phase), feat_mag * torch.sin(feat_phase)),
        #                                       dim=1)
        #     esti_x = model(feat_x_compress)
        #     esti_mag, esti_phase = torch.norm(esti_x, dim=1) ** 2.0, torch.atan2(esti_x[:, -1, :, :],
        #                                                                             esti_x[:, 0, :, :])
        #     esti_x = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
        #     esti_x = esti_x.permute(0, 3, 2, 1)
        #
        #     esti_utt = torch.istft(esti_x, 320, 160, 320, window=torch.hann_window(320).cuda(),
        #                             length=wav_len).transpose(-2, -1).squeeze()
        #     esti_utt = esti_utt.cpu().numpy()
        #     esti_utt = esti_utt[:wav_len]
        #     esti_utt = esti_utt / c
        #     os.makedirs(esti_file_path, exist_ok=True)
        #     sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
        #     print(' The %d utterance has been decoded!' % (cnt + 1))
        #     cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/datasets/VB_DEMAND_48K/noisy_16k')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/vb_taylor_cprs_model')
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    args = parser.parse_args()
    #    print(args)
    enhance(args=args)
