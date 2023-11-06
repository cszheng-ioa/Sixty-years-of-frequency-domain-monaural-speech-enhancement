import torch
import argparse
import os
import numpy as np
from ds_gagnet import DsGagnet
#from gaf_net import gaf_base
#from gagnet_cln import GaGNet
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enhance(args):
    #model = gaf_base(is_aux=False)
    model = DsGagnet(k1=[2, 3],
                     k2=[1, 3],
                     c=64,
                     intra_connect="cat",
                     kd1=3,
                     cd1=64,
                     tcm_num=2,
                     dilas=[1, 2, 5, 9],
                     fft_num=320,
                     is_causal=True,
                     acti_type="sigmoid",
                     crm_type='crm1',
                     stage_num=3,
                     u_type='u2',
                     head_type='RI+MAG')

    checkpoint = torch.load(args.Model_path)
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    # with torch.no_grad():
    #     cnt = 0
    #     mix_file_path = args.mix_file_path
    #     seen_flag = args.seen_flag
    #     noise_type = args.noise_type
    #     esti_file_path = args.esti_file_path
    #     snr = args.snr
    #
    #     for _, snr_key in enumerate(snr):
    #         if seen_flag == 1:
    #             mix_file_path1 = os.path.join(mix_file_path, noise_type, 'seen', str(snr_key))
    #         else:
    #             mix_file_path1 = os.path.join(mix_file_path, noise_type, 'unseen', str(snr_key))
    #         file_list = os.listdir(mix_file_path1)
    #         for file_id in file_list:
    #             file_name = file_id.split('.')[0]
    #             feat_wav, _ = sf.read(os.path.join(mix_file_path1, file_id))
    #             c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
    #             feat_wav = feat_wav * c
    #             wav_len = len(feat_wav)
    #             frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
    #             fake_wav_len = (frame_num - 1) * 160 + 320 - 320
    #             left_sample = fake_wav_len - wav_len
    #             feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
    #             feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
    #                                 window=torch.hann_window(320)).permute(0, 3, 2, 1).cuda()
    #
    #             feat_mag, feat_phase = torch.norm(feat_x, dim=1)**1.0, torch.atan2(feat_x[:,-1,:,:], feat_x[:,0,:,:])
    #             feat_x_compress = torch.stack((feat_mag*torch.cos(feat_phase), feat_mag*torch.sin(feat_phase)), dim=1)
    #             esti_list, glance_list = model(feat_x_compress)
    #
    #             esti_x = esti_list[2]
    #             esti_mag, esti_phase = torch.norm(esti_x, dim=1)**1.0, torch.atan2(esti_x[:,-1,:,:], esti_x[:,0,:,:])
    #             esti_x = torch.stack((esti_mag*torch.cos(esti_phase), esti_mag*torch.sin(esti_phase)), dim=1)
    #             esti_x = esti_x.permute(0,2,3,1)
    #
    #             esti_utt = torch.istft(esti_x, 320, 160, 320, window=torch.hann_window(320).cuda(), center=True).transpose(-2,-1).squeeze()
    #             esti_utt = esti_utt.cpu().numpy()
    #             esti_utt = esti_utt[:wav_len]
    #             esti_utt = esti_utt / c
    #             if seen_flag == 1:
    #                 os.makedirs(os.path.join(esti_file_path, noise_type, 'seen', str(snr_key), 'csru_stage2'), exist_ok=True)
    #                 sf.write(os.path.join(esti_file_path, noise_type, 'seen', str(snr_key), 'csru_stage2', file_id), esti_utt, args.fs)
    #             else:
    #                 os.makedirs(os.path.join(esti_file_path, noise_type, str(snr_key), 'csru_stage3'), exist_ok=True)
    #                 sf.write(os.path.join(esti_file_path, noise_type, str(snr_key), 'csru_stage3', file_id), esti_utt, args.fs)
    #             print(' The %d utterance has been decoded!' % (cnt + 1))
    #             cnt += 1

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        seen_flag = args.seen_flag
        noise_type = args.noise_type
        esti_file_path = args.esti_file_path
        snr = args.snr

        for _, snr_key in enumerate(snr):
            if seen_flag == 1:
                mix_file_path1 = os.path.join(mix_file_path, noise_type, 'seen', str(snr_key))
            else:
                mix_file_path1 = os.path.join(mix_file_path, noise_type, 'unseen', str(snr_key))
            file_list = os.listdir(mix_file_path1)
            for file_id in file_list:
                file_name = file_id.split('.')[0]
                feat_wav, _ = sf.read(os.path.join(mix_file_path1, file_id))
                c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
                feat_wav = feat_wav * c
                wav_len = len(feat_wav)
                frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
                fake_wav_len = (frame_num - 1) * 160 + 320 - 320
                left_sample = fake_wav_len - wav_len
                feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
                feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                    window=torch.hann_window(320)).permute(0, 3, 2, 1).cuda()
                feat_mag, feat_phase = torch.norm(feat_x, dim=1)**1.0, torch.atan2(feat_x[:,-1,:,:], feat_x[:,0,:,:])
                feat_x_compress = torch.stack((feat_mag*torch.cos(feat_phase), feat_mag*torch.sin(feat_phase)), dim=1)
                esti_list = model(feat_x_compress)

                esti_x = esti_list[-1]
                esti_mag, esti_phase = torch.norm(esti_x, dim=1)**1.0, torch.atan2(esti_x[:,-1,:,:], esti_x[:,0,:,:])
                esti_x = torch.stack((esti_mag*torch.cos(esti_phase), esti_mag*torch.sin(esti_phase)), dim=1)
                esti_x = esti_x.permute(0,2,3,1)

                esti_utt = torch.istft(esti_x, 320, 160, 320, window=torch.hann_window(320).cuda(), center=True).transpose(-2,-1).squeeze()
                esti_utt = esti_utt.cpu().numpy()
                esti_utt = esti_utt[:wav_len]
                esti_utt = esti_utt / c
                if seen_flag == 1:
                    os.makedirs(os.path.join(esti_file_path, noise_type, 'seen', str(snr_key)), exist_ok=True)
                    sf.write(os.path.join(esti_file_path, noise_type, 'seen', str(snr_key), file_id), esti_utt, args.fs)
                else:
                    os.makedirs(os.path.join(esti_file_path, noise_type, 'unseen', str(snr_key)), exist_ok=True)
                    sf.write(os.path.join(esti_file_path, noise_type, 'unseen', str(snr_key), file_id), esti_utt, args.fs)
                print(' The %d utterance has been decoded!' % (cnt + 1))
                cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/wsj0_si84_300h_g2net_noncprs_model')
    parser.add_argument('--noise_type', type=str, default='cafe')  # babble   cafe
    parser.add_argument('--seen_flag', type=int, default=1)    # 1   0
    parser.add_argument('--snr', type=int, default=[-5, 0, 5, 10])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/wsj0_si84_300h_g2net_noncprs_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
#    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/wsj0_si84_300h_g2net_noncprs_model')
    parser.add_argument('--noise_type', type=str, default='cafe')  # babble   cafe
    parser.add_argument('--seen_flag', type=int, default=0)    # 1   0
    parser.add_argument('--snr', type=int, default=[-5, 0, 5, 10])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/wsj0_si84_300h_g2net_noncprs_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/wsj0_si84_300h_g2net_noncprs_model')
    parser.add_argument('--noise_type', type=str, default='factory1')  # babble   cafe
    parser.add_argument('--seen_flag', type=int, default=1)  # 1   0
    parser.add_argument('--snr', type=int, default=[-5, 0, 5, 10])  # -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/wsj0_si84_300h_g2net_noncprs_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    #    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/wsj0_si84_300h_g2net_noncprs_model')
    parser.add_argument('--noise_type', type=str, default='factory1')  # babble   cafe
    parser.add_argument('--seen_flag', type=int, default=0)  # 1   0
    parser.add_argument('--snr', type=int, default=[-5, 0, 5, 10])  # -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/wsj0_si84_300h_g2net_noncprs_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/wsj0_si84_300h_g2net_noncprs_model')
    parser.add_argument('--noise_type', type=str, default='babble')  # babble   cafe
    parser.add_argument('--seen_flag', type=int, default=1)  # 1   0
    parser.add_argument('--snr', type=int, default=[-5, 0, 5, 10])  # -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/wsj0_si84_300h_g2net_noncprs_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    #    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/wsj0_si84_300h_g2net_noncprs_model')
    parser.add_argument('--noise_type', type=str, default='babble')  # babble   cafe
    parser.add_argument('--seen_flag', type=int, default=0)  # 1   0
    parser.add_argument('--snr', type=int, default=[-5, 0, 5, 10])  # -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/wsj0_si84_300h_g2net_noncprs_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    enhance(args=args)