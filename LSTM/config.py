import os

# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = int(8.0*16000)
json_dir = '/media/luoxiaoxue/datasets/VB_DEMAND_48K/json'
file_path = '/media/luoxiaoxue/datasets/VB_DEMAND_48K'
loss_dir = './LOSS/vb_lstm_cprs_loss.mat'
batch_size = 32
epochs = 100
lr = 1e-3
model_best_path = './BEST_MODEL/vb_lstm_cprs_model.pth'
check_point_path = './CP_dir'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)