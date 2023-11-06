# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 8*16000

# network parameter setings
X = 6
R = 3
alpha = 0.1

# server parameter settings
batch_size = 16
epochs = 50
lr = 1e-3

#WSJ
#json_dir = '/media/luoxiaoxue/datasets/wsj0_si84_300h/Json'
#file_path = '/media/luoxiaoxue/datasets/wsj0_si84_300h'
#VB
json_dir = '/media/luoxiaoxue/datasets/VB_DEMAND_48K/json'
file_path = '/media/luoxiaoxue/datasets/VB_DEMAND_48K'

loss_dir = './LOSS/step2_cv_cts_noncprs_model_loss.mat'
best_path = './BEST_MODEL/step2_cv_cts_noncprs_model.pth'
pretrained_path = './MODEL/step1_cv_cts_noncprs_model_tmp.pth'
# checkpoint_path = './MODEL'
model1_best_path = './BEST_MODEL/step1_cv_cts_noncprs_model_final.pth'