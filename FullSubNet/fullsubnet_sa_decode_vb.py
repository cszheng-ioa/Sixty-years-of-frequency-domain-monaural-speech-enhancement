import torch
import argparse
import librosa
import os
import numpy as np
from fullsubnet_net_sa.model import Model
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def enhance(args):
    model = Model(
        sb_num_neighbors=15,
        fb_num_neighbors=0,
        num_freqs=257,
        look_ahead=2,
        sequence_model="LSTM",
        fb_output_activate_function="ReLU",
        sb_output_activate_function=None,
        fb_model_hidden_size=512,
        sb_model_hidden_size=384,
        weight_init=True,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2,
    )
    model.load_state_dict(torch.load('./BEST_MODEL/vb_fullsubnet_noncprs_model_512_256.pth'))
    print(model)
    model.eval()
    model.cuda()

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
            frame_num = int(np.ceil((wav_len - 512 + 512) / 256 + 1))
            fake_wav_len = (frame_num - 1) * 255 + 512 - 512
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(feat_wav)
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=512, hop_length=256, win_length=512,
                                window=torch.hann_window(512)).permute(0,3,1,2).cuda()

            #cprs
            feat_x_mag = torch.norm(feat_x, dim=1) ** 1.
            feat_x_phase = torch.atan2(feat_x[:, 1, :, :], feat_x[:, 0, :, :])
            feat_x = torch.stack((feat_x_mag * torch.cos(feat_x_phase), feat_x_mag * torch.sin(feat_x_phase)), dim=1)

            feat_mag = torch.norm(feat_x, dim=1, keepdim=True)

            batch_esti_mask = model(feat_mag)
            batch_esti_mask_r, batch_esti_mask_i = batch_esti_mask[:, 0, ...], batch_esti_mask[:, -1, ...]
            batch_feat_r, batch_feat_i = feat_x[:, 0, ...], feat_x[:, -1, ...]
            batch_esti_r = batch_esti_mask_r * batch_feat_r - batch_esti_mask_i * batch_feat_i
            batch_esti_i = batch_esti_mask_r * batch_feat_i + batch_esti_mask_i * batch_feat_r
            esti_x = torch.stack((batch_esti_r, batch_esti_i), dim=1)

            #uncprs
            esti_x_mag = torch.norm(esti_x, dim=1) ** 1.
            esti_x_phase = torch.atan2(esti_x[:, 1, :, :], esti_x[:, 0, :, :])
            esti_x = torch.stack((esti_x_mag * torch.cos(esti_x_phase), esti_x_mag * torch.sin(esti_x_phase)), dim=1)

            esti_x = esti_x.permute(0,2,3,1).contiguous().cpu()
            esti_utt = torch.istft(esti_x, 512, 256, 512, torch.hann_window(512), length=wav_len).numpy()
            esti_utt = np.squeeze(esti_utt[:wav_len])
            esti_utt = esti_utt / c

            os.makedirs(esti_file_path, exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)

            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/media/luoxiaoxue/datasets/VB_DEMAND_48K/noisy_16k')
    parser.add_argument('--esti_file_path', type=str, default='./datasets/vb_fullsubnet_noncprs_model')
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    args = parser.parse_args()
#    print(args)
    enhance(args=args)

    enhance(args=args)