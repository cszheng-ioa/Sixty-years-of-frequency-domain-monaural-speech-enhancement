import os
# for complex network baseline
# front-end parameter settings
fs = 16000
win_size = 512
fft_num = 512
win_shift = 128
chunk_length = 8*16000



#WSJ
#json_dir = '/media/luoxiaoxue/datasets/wsj0_si84_300h/Json'
#file_path = '/media/luoxiaoxue/datasets/wsj0_si84_300h'
#VB
json_dir = '/media/luoxiaoxue/datasets/wsj0_si84_300h/Json'
file_path = '/media/luoxiaoxue/datasets/wsj0_si84_300h'
loss_dir = './LOSS/wsj0_si84_300h_dccrn_snr_model_loss.mat'
cpk_path = './ckp_model_dir'
is_cpk = True
batch_size = 16
epochs = 60
lr = 1e-3
best_path = './BEST_MODEL/wsj0_si84_300h_dccrn_snr_model.pth'

os.makedirs('./LOSS', exist_ok=True)
os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs(cpk_path, exist_ok=True)
