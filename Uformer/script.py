from trans import STFT, iSTFT, MelTransform
import soundfile as sf
import torch
from scipy import signal
import numpy as np 

stft = STFT(400, 160)
istft = iSTFT(400, 160)

y1, fs = sf.read('../fuck/1089-134686-0000.flac')
y2, fs = sf.read('../fuck/121-121726-0000.flac')
rir, fs = sf.read('../fuck/5.46_4.40_3.58_3.04_1.97_1.36_218.3015_254.7778_26.3611_0.2320.wav')

if y1.shape[0] > y2.shape[0]:
	length = y2.shape[0]
else:
	length = y1.shape[0]

y1 = y1[:length]
y2 = y2[:length]
a = []
b = []
for i in range(8):
	a.append(signal.oaconvolve(y1, rir[:, i]))
	b.append(signal.oaconvolve(y2, rir[:, i+8]))
y1 = np.stack(a, -1)
y2 = np.stack(b, -1)
y1 = y1[:-7999]
y2 = y2[:-7999]
a=y1
mixwav = y1 + y2
y1 = torch.from_numpy(y1).float()
y2 = torch.from_numpy(y2).float()
y1 = y1.unsqueeze(0).transpose(2, 1)
y2 = y2.unsqueeze(0).transpose(2, 1)
mix = y1 + y2
mix_real, mix_imag = stft(mix)
y1_real, y1_imag = stft(y1)
y2_real, y2_imag = stft(y2)

mix_mag = torch.sqrt(torch.clamp(mix_real**2 + mix_imag**2, 1e-7))
y1_mag = torch.sqrt(torch.clamp(y1_real**2 + y1_imag**2, 1e-7))
y2_mag = torch.sqrt(torch.clamp(y2_real**2 + y2_imag**2, 1e-7))
mix_pha = torch.atan2(mix_imag, mix_real)
mask = y1_mag / torch.clamp(mix_mag, 1e-7)
mask = mask[:,0]
# max_abs = torch.norm(mask, float("inf"), dim=1, keepdim=True)
# mask = mask / torch.clamp(max_abs, 1e-7)
# mask = torch.transpose(mask, 1, 2)
mask = mask.squeeze()

mask = np.array(mask)
enh = mix_mag[0,0]*mask#.transpose(1,0)
enh = enh.unsqueeze(0)


enh = istft((enh*torch.cos(mix_pha[:,0]), enh*torch.sin(mix_pha[:,0])))
enh = enh.squeeze()
enh = np.array(enh)
sf.write('../fuck/masked.wav',enh,16000)
print(enh.shape)



np.save('../fuck/testmask.npy', mask.transpose(1,0))
sf.write('../fuck/mix.wav',mixwav, 16000)
sf.write('../fuck/spk1.wav',a, 16000)