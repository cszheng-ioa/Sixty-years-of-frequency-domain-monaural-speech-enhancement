win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 8*16000

json_dir = '/media/luoxiaoxue/datasets/VB_DEMAND_48K/json'
file_path = '/media/luoxiaoxue/datasets/VB_DEMAND_48K'
# the file name of the training and validation loss
loss_dir = './LOSS/vb_gaf_cprs_loss.mat'
batch_size = 16
epochs = 50
lr = 1e-3
best_path = './BEST_MODEL/vb_gaf_cprs_model.pth'
checkpoint_path = './MODEL'