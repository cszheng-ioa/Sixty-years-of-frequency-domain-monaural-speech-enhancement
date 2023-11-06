import torch
import argparse
import librosa
import os
import numpy as np
import json
import scipy
from Backup import *
import pickle
from config import *
from CRN import crn_net
# from Step1_network_id_wi_ss import Step1_net
# from LSTM import lstm_net
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enhance(args):
    model = crn_net()
    model.load_state_dict(torch.load('./BEST_MODEL/wsj0_si84_300h_crn_noncprs_model.pth'))
    # print(model)
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
        if not os.path.exists(esti_file_path):
            os.makedirs(esti_file_path)

        file_list = os.listdir(mix_file_path)
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

            #compressed
            feat_x, phase_x = np.abs(feat_x)**1.0, np.angle(feat_x)

            feat_x = torch.FloatTensor(feat_x)
            feat_x = feat_x.cuda()
            esti_x = model(feat_x.unsqueeze(dim=0))

            #compressed
            esti_x = esti_x ** 1.

            esti_x = esti_x.squeeze(dim=0).cpu().numpy()
            de_esti = np.multiply(esti_x, np.exp(1j * phase_x))
            esti_utt = librosa.istft((de_esti).T, hop_length=win_shift,
                                     win_length=win_size, window='hanning', length=len(feat_wav))
            esti_utt = esti_utt / c
            # if seen_flag == 1:
            #     os.makedirs(os.path.join(esti_file_path, noise_type, 'seen', str(snr)), exist_ok=True)
            #     librosa.output.write_wav(os.path.join(esti_file_path, noise_type, 'seen', str(snr), file_id), esti_utt,
            #                              args.fs)
            # else:
            #     os.makedirs(os.path.join(esti_file_path, noise_type, 'unseen', str(snr)), exist_ok=True)
            #     librosa.output.write_wav(os.path.join(esti_file_path, noise_type, 'unseen', str(snr), file_id), esti_utt,
            #                              args.fs)
            # os.makedirs(os.path.join(esti_file_path, str(snr)), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)

    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='cafe')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='factory1')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='seen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='-5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='0')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='5')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)

    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix')
    parser.add_argument('--noise_type', type=str, default='babble')
    parser.add_argument('--seen', type=str, default='unseen')
    parser.add_argument('--snr', type=str, default='10')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/wsj0_si84_300h_crn_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()
    print(args)
    enhance(args=args)




#    parser = argparse.ArgumentParser('Recovering audio')
#    parser.add_argument('--mix_file_path', type=str,
#                        default='/media/liuwenzhe/datasets/WSJ-SI-84-Trans/test_vl/')
#    parser.add_argument('--noise_type', type=str, default='factory1')
#    parser.add_argument('--seen', type=str, default='seen')
#    parser.add_argument('--snr', type=str, default='0')
#    parser.add_argument('--esti_clean_file_path', type=str,
#                        default='/media/liuwenzhe/datasets/WSJ-SI-84-Trans/crn_cprs_esti/')
#    parser.add_argument('--fs', type=int, default=16000)

#    args = parser.parse_args()
#    print(args)
#    enhance(args=args)
