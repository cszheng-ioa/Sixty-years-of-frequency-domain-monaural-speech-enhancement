# MIN_SNR=-10
# MAX_SNR=20
# SNR_INTER=1
# VER = 'mhanet-1.1c'
# network_type     ='MHANetV3'
# d_model           =256
# n_blocks          =5
# n_heads           =8
# warmup_steps      =40000
# causal            =1
# max_len           =2048
# loss_fnc          ="BinaryCrossentropy"
# outp_act          ="Sigmoid"
# max_epochs        =50
# resume_epoch      =0
# test_epoch        =50
# mbatch_size       =4
# inp_tgt_type      ='MagXi'
# map_type          ='DBNormalCDF'
# sample_size       =1000
# f_s               =16000
# T_d               =32
# T_s               =16
# # min_snr           =MIN_SNR
# # max_snr           =MAX_SNR
# # snr_inter         =SNR_INTER
# # out_type          =OUT_TYPE
# save_model        =1
# log_iter          =0
# eval_example      =1
# gain              ='mmse-lsa'
# train             =1
# infer             =0
# test              =0
# gpu               ='3'




VER = 'resnet-1.1c-wsj'
network_type     ='ResNetV2'
d_model           =256
n_blocks          =40
d_f = 64
k = 3
max_d_rate = 16
causal = 1
unit_type = "ReLU->LN->W+b"
loss_fnc = "BinaryCrossentropy"
outp_act = "Sigmoid"
max_epochs = 50
resume_epoch = 0
test_epoch = 50
mbatch_size = 8
inp_tgt_type = "MagXi"
map_type = 'DBNormalCDF'
sample_size = 1000
f_s = 16000
T_d = 32
T_s = 16
save_model = 1
log_iter = 0
eval_example = 1
gain              ='mmse-lsa'
train             =0
infer             =1
test              =0
gpu               ='3'
out_type = 'y'





# LOG_PATH = 'log'
# SET_PATH = '/home/aaron/set/deep_xi_dataset'
# DATA_PATH = '/home/aaron/mnt/fist/data/'$PROJ_DIR
# TEST_X_PATH = '/home/aaron/mnt/aaron/set/deep_xi_dataset/test_noisy_speech'
# TEST_S_PATH = '/home/aaron/mnt/aaron/set/deep_xi_dataset/test_clean_speech'
# TEST_D_PATH = '/home/aaron/mnt/aaron/set/deep_xi_dataset/test_noise'
# OUT_PATH = '/home/aaron/mnt/aaron_root/mnt/hdd1/out/'$PROJ_DIR
# MODEL_PATH = '/home/aaron/mnt/fist/model/'$PROJ_DIR
LOG_PATH = './log'
##SET_PATH = '/media/dailinlin/dataset/wsj0_si84_300h'
# SET_PATH = '/media/liandong/wsj0_si84_300h'
DATA_PATH = './data'
OUT_PATH = './out'
MODEL_PATH = './model'
#TEST_X_PATH = '/media/dailinlin/dataset/wsj0_si84_300h/dev/mix'
#TEST_S_PATH = '/media/dailinlin/dataset/wsj0_si84_300h/dev/clean'

SET_PATH = '/media/luoxiaoxue/datasets/wsj0_si84_300h'
TEST_X_PATH = '/media/luoxiaoxue/dataset/wsj0_si84_300h/cv/mix'
TEST_S_PATH = '/media/luoxiaoxue/dataset/wsj0_si84_300h/cv/clean'


#SET_PATH = '/media/luoxiaoxue/datasets/VB_DEMAND_48K'
#TEST_S_PATH = '/media/luoxiaoxue/datasets/VB_DEMAND_48K/clean_testset_wav_16k'
#TEST_X_PATH = '/media/luoxiaoxue/datasets/VB_DEMAND_48K/noisy_testset_wav_16k'


set_path          =SET_PATH
data_path         =DATA_PATH
log_path = LOG_PATH
test_x_path       =TEST_X_PATH
test_s_path       =TEST_S_PATH
# test_d_path       =TEST_D_PATH
out_path          =OUT_PATH
model_path        =MODEL_PATH

max_wav_len = 8 * 16000