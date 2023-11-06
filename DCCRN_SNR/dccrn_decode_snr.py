import torch
import argparse
import librosa
import os
import numpy as np
from istft import ISTFT
from DCCRN import DCCRN
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enhance(args):
    model = DCCRN(rnn_units=256, use_clstm=True, kernel_num=[32, 64, 128, 256, 256, 256]).cuda()
    model.load_state_dict(torch.load('./BEST_MODEL/wsj0_si84_300h_dccrn_snr_model.pth'))
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_clean_file_path
        seen_flag = args.seen
        noise_type = args.noise_type
        snr = args.snr
        mix_file_path = os.path.join(mix_file_path, noise_type, seen_flag, snr)
        esti_file_path = os.path.join(esti_file_path, noise_type, seen_flag, snr)
        os.makedirs(esti_file_path, exist_ok=True)

        file_list = os.listdir(mix_file_path)
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            # ri+mag的设置的窗长是512,帧移是128，FFT是512
            # snr的窗长是400,帧移是100，FFT是512
            frame_num = int(np.ceil((wav_len - 512 + 512) / 128 + 1))
            fake_wav_len = (frame_num - 1) * 128 + 512 - 512
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
#            feat_wav = torch.FloatTensor(feat_wav)
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=512, hop_length=128, win_length=512,
                                window=torch.hann_window(512)).permute(0, 3, 1, 2)# B 2 F T
            # compress
            feat_x, phase_x = torch.norm(feat_x, dim=1) ** 1., torch.atan2(feat_x[:, 1, :, :], feat_x[:, 0, :, :])

            feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=1)

            esti_x = model(feat_x.cuda()).cpu()# B 2 T F
            esti_x_mag = torch.norm(esti_x, dim=1)
            esti_x_phase = torch.atan2(esti_x[:, 1, :, :], esti_x[:, 0, :, :])
            # compressed
            esti_x_mag = esti_x_mag ** 1.

            esti_x = torch.stack((esti_x_mag * torch.cos(esti_x_phase), esti_x_mag * torch.sin(esti_x_phase)), dim=1)
#            esti_x = esti_x.permute(0, 3, 2, 1)
            esti_x = esti_x.permute(0, 2, 3, 1)
            esti_utt = torch.istft(esti_x, 512, 128, 512, torch.hann_window(512)) # b F T 2

##            esti_x_mag = esti_x_mag.squeeze(dim=0).numpy()
##            esti_x_phase = esti_x_phase.squeeze(dim=0).numpy()
##            de_esti = np.multiply(esti_x_mag, np.exp(1j * esti_x_phase))
##            esti_utt = librosa.istft((de_esti).T, hop_length=128,
##                                     win_length=512, window='hanning', length=len(feat_wav))
            esti_utt = esti_utt.squeeze(dim=0)
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c

            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

    # with torch.no_grad():
    #     cnt = 0
    #     mix_file_path = args.mix_file_path
    #     esti_file_path = args.esti_file_path
    #     seen_flag = args.seen_flag
    #     noise_type = args.noise_type
    #     snr = args.snr
    #     for _, snr_key in enumerate(snr):
    #         if seen_flag == 1:
    #             mix_file_path1 = os.path.join(mix_file_path, noise_type, 'seen', str(snr_key))
    #         else:
    #             mix_file_path1 = os.path.join(mix_file_path, noise_type, 'unseen', str(snr_key))
    #         #mix_file_path1 = os.path.join(mix_file_path, str(snr_key))
    #         file_list = os.listdir(mix_file_path1)
    #         for file_id in file_list:
    #             feat_wav, _ = sf.read(os.path.join(mix_file_path1, file_id))
    #             c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
    #             feat_wav = feat_wav * c
    #             wav_len = len(feat_wav)
    #             frame_num = int(np.ceil((wav_len - 512 + 512) / 128 + 1))
    #             fake_wav_len = (frame_num - 1) * 128 + 512 - 512
    #             left_sample = fake_wav_len - wav_len
    #             feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
    #             feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=512, hop_length=128, win_length=512,
    #                                 window=torch.hann_window(512)).permute(0, 3, 1, 2)
    #             esti_x = model(feat_x.cuda()).cpu().permute(0,3,2,1)
    #             #esti_utt = istft(esti_x).squeeze().numpy()
    #             esti_utt = torch.istft(esti_x, 512, 128, 512, window=torch.hann_window(512)).squeeze(dim=0)
    #             esti_utt = esti_utt[:wav_len]
    #             esti_utt = esti_utt / c
    #             if seen_flag == 1:
    #                 os.makedirs(os.path.join(esti_file_path, noise_type, 'seen', str(snr_key)), exist_ok=True)
    #                 sf.write(os.path.join(esti_file_path, noise_type, 'seen', str(snr_key), file_id), esti_utt, args.fs)
    #             else:
    #                 os.makedirs(os.path.join(esti_file_path, noise_type, 'unseen', str(snr_key)), exist_ok=True)
    #                 sf.write(os.path.join(esti_file_path, noise_type, 'unseen', str(snr_key), file_id), esti_utt, args.fs)
    #             # os.makedirs(os.path.join(esti_file_path,  str(snr_key)), exist_ok=True)
    #             # sf.write(os.path.join(esti_file_path, str(snr_key), file_id), esti_utt, args.fs)
    #             print(' The %d utterance has been decoded!' % (cnt + 1))
    #             cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_dccrn_snr_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    enhance(args=args)
