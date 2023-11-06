## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from deepxi.args_resnet import get_args
from deepxi.model import DeepXi
from deepxi.prelim import Prelim
from deepxi.se_batch import Batch_test
import deepxi.utils as utils
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config0 = ConfigProto()
# config0.gpu_options.allow_growth = True
# session = InteractiveSession(config=config0)



if __name__ == '__main__':

	args = get_args()

	print("Arguments:")
	[print(key,val) for key,val in vars(args).items()]

	if args.causal: args.padding = "causal"
	else: args.padding = "same"

	args.model_path = args.model_path + '/' + args.ver # model save path.
	# if args.set_path != "set": args.data_path = args.data_path + '/' + args.set_path.rsplit('/', 1)[-1] # data path.
	train_s_path = args.set_path + '/train/clean' # path to the clean speech training set.
	train_x_path = args.set_path + '/train/mix' # path to the noise training set.
	val_s_path = args.set_path + '/dev/clean' # path to the clean speech validation set.
	val_x_path = args.set_path + '/dev/mix' # path to the noise validation set.
	N_d = int(args.f_s*args.T_d*0.001) # window duration (samples).
	N_s = int(args.f_s*args.T_s*0.001) # window shift (samples).
	K = int(pow(2, np.ceil(np.log2(N_d)))) # number of DFT components.

	# if args.train:
	# 	train_s_list = utils.batch_list(train_s_path, 'clean_speech', args.data_path)
	# 	train_d_list = utils.batch_list(train_x_path, 'noise', args.data_path)
	# 	if args.val_flag:
	# 		val_s, val_d, val_s_len, val_d_len, val_snr = utils.val_wav_batch(val_s_path, val_x_path)
	# 	else: val_s, val_d, val_s_len, val_d_len, val_snr = None, None, None, None, None
	# else: train_s_list, train_d_list = None, None

	args.train = False
	args.infer = True
	args.test = False
	args.max_epochs =  50
	args.test_epoch = 50
	test_choice = 'babble/unseen'
	args.test_x_path = os.path.join('/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix', test_choice)
	args.test_s_path = '/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/datasets/mix'
	args.out_path = os.path.join('/media/luoxiaoxue/LXX2/denoise_review/speech_enhancement_overview/DeepXi_dll/out/y/WSJ/', test_choice)
	train_json_path = args.set_path + '/Json/train/mix_files.json'
	val_json_path = args.set_path + '/Json/dev/mix_files.json'
	os.makedirs(args.out_path, exist_ok=True)


	config = utils.gpu_config(args.gpu)

	print("Version: %s." % (args.ver))

	deepxi = DeepXi(
		N_d=N_d,
		N_s=N_s,
		K=K,
		sample_dir=args.data_path,
		train_s_list=None,
		train_x_list=None,
		**vars(args)
		)

	if args.infer:
		snrs = [-5, 0, 5, 10]
		for snr in snrs:
			print('snr:', str(snr), 'dB', 'processing-------------')
			test_x_path_snr = os.path.join(	args.test_x_path, str(snr) )
			if args.infer or args.test:
				print(test_x_path_snr)
				test_x, test_x_len, _, test_x_base_names = Batch_test(test_x_path_snr)
			out_path_snr = os.path.join(args.out_path, str(snr))
			os.makedirs(out_path_snr, exist_ok=True)
			print(args.gain)
			deepxi.infer(
			test_x=test_x,
			test_x_len=test_x_len,
			test_x_base_names=test_x_base_names,
			test_epoch=args.test_epoch,
			model_path=args.model_path,
			out_type=args.out_type,
			gain=args.gain,
			out_path=out_path_snr,
			n_filters=args.n_filters,
			saved_data_path=args.saved_data_path,
			)


