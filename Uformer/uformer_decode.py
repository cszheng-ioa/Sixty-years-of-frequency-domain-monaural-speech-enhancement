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
#from GCRN_noncprs import Net
from uformer import Uformer #_uncprs
#from Step1_network_id_wi_ss import Step1_net
# from LSTM import lstm_net
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def enhance(args):
    model = Uformer()
    model.load_state_dict(torch.load('./BEST_MODEL/wsj0_si84_300h_uformer_noncprs_model.pth'))
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
            #            feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

            feat_x = torch.FloatTensor(feat_wav).cuda()
            esti_utt, src, output_cplx, src_cplx = model(feat_x.unsqueeze(dim=0), feat_x.unsqueeze(dim=0))

            esti_utt = esti_utt.squeeze(dim=0).cpu().numpy()
            esti_utt = esti_utt / c

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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
                        default='./datasets/wsj0_si84_300h_uformer_noncprs_model_new')
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
