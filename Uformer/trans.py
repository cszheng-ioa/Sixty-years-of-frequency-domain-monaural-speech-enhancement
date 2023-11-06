# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as tf
import librosa.filters as filters
import soundfile as sf
import librosa
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]) + '/model')

EPSILON = 1e-5
# from typing import Optional, Union, Tuple


def init_window(wnd: str, frame_len: int) -> th.Tensor:
    """
    Return window coefficient
    Args:
        wnd: window name
        frame_len: length of the frame
    """

    def sqrthann(frame_len, periodic=True):
        return th.hann_window(frame_len, periodic=periodic)**0.5

    if wnd not in ["bartlett", "hann", "hamm", "blackman", "rect", "sqrthann"]:
        raise RuntimeError(f"Unknown window type: {wnd}")

    wnd_tpl = {
        "sqrthann": sqrthann,
        "hann": th.hann_window,
        "hamm": th.hamming_window,
        "blackman": th.blackman_window,
        "bartlett": th.bartlett_window,
        "rect": th.ones
    }
    if wnd != "rect":
        # match with librosa
        c = wnd_tpl[wnd](frame_len, periodic=True)
    else:
        c = wnd_tpl[wnd](frame_len)
    return c


def init_kernel(frame_len: int,
                frame_hop: int,
                window: str,
                round_pow_of_two: bool = True,
                normalized: bool = False,
                inverse: bool = False,
                mode: str = "librosa") -> th.Tensor:
    """
    Return STFT kernels
    Args:
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window name
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        normalized: return normalized DFT matrix
        inverse: return iDFT matrix
        mode: framing mode (librosa or kaldi)
    """
    if mode not in ["librosa", "kaldi"]:
        raise ValueError(f"Unsupported mode: {mode}")
    # FFT points
    B = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # center padding window if needed
    if mode == "librosa" and B != frame_len:
        lpad = (B - frame_len) // 2
        window = tf.pad(window, (lpad, B - frame_len - lpad))
    if normalized:
        # make K^H * K = I
        S = B**0.5
    else:
        S = 1
    I = th.stack([th.eye(B), th.zeros(B, B)], dim=-1)
    # W x B x 2
    K = th.fft.fft(I / S, 1)
    K = th.cat([K.real, K.imag], -1)
    if mode == "kaldi":
        K = K[:frame_len]
    if inverse and not normalized:
        # to make K^H * K = I
        K = K / B
    # 2 x B x W
    K = th.transpose(K, 0, 2) * window
    # 2B x 1 x W
    K = th.reshape(K, (B * 2, 1, K.shape[-1]))
    return K, window


def mel_filter(frame_len: int,
               round_pow_of_two: bool = True,
               num_bins: int = None,
               sr: int = 16000,
               num_mels: int = 80,
               fmin: float = 0.0,
               fmax: float = None,
               norm: bool = False) -> th.Tensor:
    """
    Return mel filter coefficients
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        num_bins: number of the frequency bins produced by STFT
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        norm: normalize the mel filter coefficients
    """
    # FFT points
    if num_bins is None:
        N = 2**math.ceil(
            math.log2(frame_len)) if round_pow_of_two else frame_len
    else:
        N = (num_bins - 1) * 2
    # fmin & fmax
    freq_upper = sr // 2
    if fmax is None:
        fmax = freq_upper
    else:
        fmax = min(fmax + freq_upper if fmax < 0 else fmax, freq_upper)
    fmin = max(0, fmin)
    # mel filter coefficients
    mel = filters.mel(sr,
                      N,
                      n_mels=num_mels,
                      fmax=fmax,
                      fmin=fmin,
                      htk=True,
                      norm="slaney" if norm else None)
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)

def inv_mel_filter(frame_len: int,
                   round_pow_of_two: bool = True,
                   num_bins: int = None,
                   sr: int = 16000,
                   num_mels: int = 80,
                   fmin: float = 0.0,
                   fmax: float = None,
                   norm: bool = False) -> th.Tensor:
    """
    Return mel filter coefficients
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        num_bins: number of the frequency bins produced by STFT
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        norm: normalize the mel filter coefficients
    """
    # FFT points
    if num_bins is None:
        N = 2**math.ceil(
            math.log2(frame_len)) if round_pow_of_two else frame_len
    else:
        N = (num_bins - 1) * 2
    # fmin & fmax
    freq_upper = sr // 2
    if fmax is None:
        fmax = freq_upper
    else:
        fmax = min(fmax + freq_upper if fmax < 0 else fmax, freq_upper)
    fmin = max(0, fmin)
    # mel filter coefficients
    mel = filters.mel(sr,
                      N,
                      n_mels=num_mels,
                      fmax=fmax,
                      fmin=fmin,
                      htk=True,
                      norm="slaney" if norm else None)
    mel = np.linalg.pinv(mel)
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)


def speed_perturb_filter(src_sr: int,
                         dst_sr: int,
                         cutoff_ratio: float = 0.95,
                         num_zeros: int = 64) -> th.Tensor:
    """
    Return speed perturb filters, reference:
        https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
    Args:
        src_sr: sample rate of the source signal
        dst_sr: sample rate of the target signal
    Return:
        weight (Tensor): coefficients of the filter
    """
    if src_sr == dst_sr:
        raise ValueError(
            f"src_sr should not be equal to dst_sr: {src_sr}/{dst_sr}")
    gcd = math.gcd(src_sr, dst_sr)
    src_sr = src_sr // gcd
    dst_sr = dst_sr // gcd
    if src_sr == 1 or dst_sr == 1:
        raise ValueError("do not support integer downsample/upsample")
    zeros_per_block = min(src_sr, dst_sr) * cutoff_ratio
    padding = 1 + int(num_zeros / zeros_per_block)
    # dst_sr x src_sr x K
    times = (np.arange(dst_sr)[:, None, None] / float(dst_sr) -
             np.arange(src_sr)[None, :, None] / float(src_sr) -
             np.arange(2 * padding + 1)[None, None, :] + padding)
    window = np.heaviside(1 - np.abs(times / padding),
                          0.0) * (0.5 + 0.5 * np.cos(times / padding * math.pi))
    weight = np.sinc(
        times * zeros_per_block) * window * zeros_per_block / float(src_sr)
    return th.tensor(weight, dtype=th.float32)


def splice_feature(feats: th.Tensor,
                   lctx: int = 1,
                   rctx: int = 1,
                   subsampling_factor: int = 1,
                   op: str = "cat") -> th.Tensor:
    """
    Splice feature
    Args:
        feats (Tensor): N x ... x T x F, original feature
        lctx: left context
        rctx: right context
        subsampling_factor: subsampling factor
        op: operator on feature context
    Return:
        splice (Tensor): feature with context padded
    """
    if lctx + rctx == 0:
        return feats
    if op not in ["cat", "stack"]:
        raise ValueError(f"Unknown op for feature splicing: {op}")
    # [N x ... x T x F, ...]
    ctx = []
    T = feats.shape[-2]
    T = T - T % subsampling_factor
    for c in range(-lctx, rctx + 1):
        idx = th.arange(c, c + T, device=feats.device, dtype=th.int64)
        idx = th.clamp(idx, min=0, max=T - 1)
        ctx.append(th.index_select(feats, -2, idx))
    if op == "cat":
        # N x ... x T x FD
        splice = th.cat(ctx, -1)
    else:
        # N x ... x T x F x D
        splice = th.stack(ctx, -1)
    return splice


def _forward_stft(
        wav: th.Tensor,
        kernel: th.Tensor,
        output: str = "polar",
        pre_emphasis: float = 0,
        frame_hop: int = 256,
        onesided: bool = False,
        center: bool = False):
    """
    STFT inner function
    Args:
        wav (Tensor), N x (C) x S
        kernel (Tensor), STFT transform kernels, from init_kernel(...)
        output (str), output format:
            polar: return (magnitude, phase) pair
            complex: return (real, imag) pair
            real: return [real; imag] Tensor
        frame_hop: frame hop size in number samples
        pre_emphasis: factor of preemphasis
        onesided: return half FFT bins
        center: if true, we assumed to have centered frames
    Return:
        transform (Tensor or [Tensor, Tensor]), STFT transform results
    """
    wav_dim = wav.dim()
    if output not in ["polar", "complex", "real"]:
        raise ValueError(f"Unknown output format: {output}")
    if wav_dim not in [2, 3]:
        raise RuntimeError(f"STFT expect 2D/3D tensor, but got {wav_dim:d}D")
    # if N x S, reshape N x 1 x S
    # else: reshape NC x 1 x S
    N, S = wav.shape[0], wav.shape[-1]
    wav = wav.contiguous().view(-1, 1, S)
    # NC x 1 x S+2P
    if center:
        pad = kernel.shape[-1] // 2
        # NOTE: match with librosa
        wav = tf.pad(wav, (pad, pad), mode="reflect")
    # STFT
    if pre_emphasis > 0:
        # NC x W x T
        frames = tf.unfold(wav[:, None], (1, kernel.shape[-1]),
                           stride=frame_hop,
                           padding=0)
        frames[:, 1:] = frames[:, 1:] - pre_emphasis * frames[:, :-1]
        # 1 x 2B x W, NC x W x T,  NC x 2B x T
        packed = th.matmul(kernel[:, 0][None, ...], frames)
    else:
        packed = tf.conv1d(wav, kernel, stride=frame_hop, padding=0)
    # NC x 2B x T => N x C x 2B x T
    if wav_dim == 3:
        packed = packed.contiguous().view(N, -1, packed.shape[-2], packed.shape[-1])
    # N x (C) x B x T
    real, imag = th.chunk(packed, 2, dim=-2)
    # N x (C) x B/2+1 x T
    if onesided:
        num_bins = kernel.shape[0] // 4 + 1
        real = real[..., :num_bins, :]
        imag = imag[..., :num_bins, :]
    if output == "complex":
        return (real, imag)
    elif output == "real":
        return th.stack([real, imag], dim=-1)
    else:
        mag = (real**2 + imag**2 + EPSILON)**0.5
        pha = th.atan2(imag, real)
        return (mag, pha)


def _inverse_stft(transform,
                  kernel: th.Tensor,
                  window: th.Tensor,
                  input: str = "polar",
                  frame_hop: int = 256,
                  onesided: bool = False,
                  center: bool = False) -> th.Tensor:
    """
    iSTFT inner function
    Args:
        transform (Tensor or [Tensor, Tensor]), STFT transform results
        kernel (Tensor), STFT transform kernels, from init_kernel(...)
        input (str), input format:
            polar: return (magnitude, phase) pair
            complex: return (real, imag) pair
            real: return [real; imag] Tensor
        frame_hop: frame hop size in number samples
        onesided: return half FFT bins
        center: used in _forward_stft
    Return:
        wav (Tensor), N x S
    """
    if input not in ["polar", "complex", "real"]:
        raise ValueError(f"Unknown output format: {input}")

    if input == "real":
        real, imag = transform[..., 0], transform[..., 1]
    elif input == "polar":
        real = transform[0] * th.cos(transform[1])
        imag = transform[0] * th.sin(transform[1])
    else:
        real, imag = transform

    # (N) x F x T
    imag_dim = imag.dim()
    if imag_dim not in [2, 3]:
        raise RuntimeError(f"Expect 2D/3D tensor, but got {imag_dim}D")

    # if F x T, reshape 1 x F x T
    if imag_dim == 2:
        real = th.unsqueeze(real, 0)
        imag = th.unsqueeze(imag, 0)

    if onesided:
        # [self.num_bins - 2, ..., 1]
        reverse = range(kernel.shape[0] // 4 - 1, 0, -1)
        # extend matrix: N x B x T
        real = th.cat([real, real[:, reverse]], 1)
        imag = th.cat([imag, -imag[:, reverse]], 1)
    # pack: N x 2B x T
    packed = th.cat([real, imag], dim=1)
    # N x 1 x T
    s = tf.conv_transpose1d(packed, kernel, stride=frame_hop, padding=0)
    # normalized audio samples
    # refer: https://github.com/pytorch/audio/blob/2ebbbf511fb1e6c47b59fd32ad7e66023fa0dff1/torchaudio/functional.py#L171
    # 1 x W x T
    win = th.repeat_interleave(window[None, ..., None],
                               packed.shape[-1],
                               dim=-1)
    # W x 1 x W
    I = th.eye(window.shape[0], device=win.device)[:, None]
    # 1 x 1 x T
    norm = tf.conv_transpose1d(win**2, I, stride=frame_hop, padding=0)
    if center:
        pad = kernel.shape[-1] // 2
        s = s[..., pad:-pad]
        norm = norm[..., pad:-pad]
    s = s / (norm + EPSILON)
    # N x S
    s = s.squeeze(1)
    return s


def forward_stft(
        wav: th.Tensor,
        frame_len: int,
        frame_hop: int,
        output: str = "complex",
        window: str = "sqrthann",
        round_pow_of_two: bool = True,
        pre_emphasis: float = 0,
        normalized: bool = False,
        onesided: bool = True,
        center: bool = False,
        mode: str = "librosa"):
    """
    STFT function implementation, equals to STFT layer
    Args:
        wav: source audio signal
        frame_len: length of the frame
        frame_hop: hop size between frames
        output: output type (complex, real, polar)
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        pre_emphasis: factor of preemphasis
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
        inverse: using iDFT kernel (for iSTFT)
        mode: "kaldi"|"librosa", slight difference on applying window function
    """
    K, _ = init_kernel(frame_len,
                       frame_hop,
                       init_window(window, frame_len),
                       round_pow_of_two=round_pow_of_two,
                       normalized=normalized,
                       inverse=False,
                       mode=mode)
    return _forward_stft(wav,
                         K.to(wav.device),
                         output=output,
                         frame_hop=frame_hop,
                         pre_emphasis=pre_emphasis,
                         onesided=onesided,
                         center=center)


def inverse_stft(transform,
                 frame_len: int,
                 frame_hop: int,
                 input: str = "complex",
                 window: str = "sqrthann",
                 round_pow_of_two: bool = True,
                 normalized: bool = False,
                 onesided: bool = True,
                 center: bool = False,
                 mode: str = "librosa") -> th.Tensor:
    """
    iSTFT function implementation, equals to iSTFT layer
    Args:
        transform: results of STFT
        frame_len: length of the frame
        frame_hop: hop size between frames
        input: input format (complex, real, polar)
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
        mode: "kaldi"|"librosa", slight difference on applying window function
    """
    if isinstance(transform, th.Tensor):
        device = transform.device
    else:
        device = transform[0].device
    K, w = init_kernel(frame_len,
                       frame_hop,
                       init_window(window, frame_len),
                       round_pow_of_two=round_pow_of_two,
                       normalized=normalized,
                       inverse=True,
                       mode=mode)
    return _inverse_stft(transform,
                         K.to(device),
                         w.to(device),
                         input=input,
                         frame_hop=frame_hop,
                         onesided=onesided,
                         center=center)


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT

    Args:
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        normalized: use normalized DFT kernel
        pre_emphasis: factor of preemphasis
        mode: "kaldi"|"librosa", slight difference on applying window function
        onesided: output onesided STFT
        inverse: using iDFT kernel (for iSTFT)
    """

    def __init__(self,
                 frame_len: int,
                 frame_hop: int,
                 window: str = "sqrthann",
                 round_pow_of_two: bool = True,
                 normalized: bool = False,
                 pre_emphasis: float = 0,
                 onesided: bool = True,
                 inverse: bool = False,
                 center: bool = False,
                 mode="librosa") -> None:
        super(STFTBase, self).__init__()
        K, w = init_kernel(frame_len,
                           frame_hop,
                           init_window(window, frame_len),
                           round_pow_of_two=round_pow_of_two,
                           normalized=normalized,
                           inverse=inverse,
                           mode=mode)
        self.K = nn.Parameter(K, requires_grad=False)
        self.w = nn.Parameter(w, requires_grad=False)
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.onesided = onesided
        self.pre_emphasis = pre_emphasis
        self.center = center
        self.mode = mode
        self.num_bins = self.K.shape[0] // 4 + 1
        self.expr = (
            f"window={window}, stride={frame_hop}, onesided={onesided}, " +
            f"pre_emphasis={self.pre_emphasis}, normalized={normalized}, " +
            f"center={self.center}, mode={self.mode}, " +
            f"kernel_size={self.num_bins}x{self.K.shape[2]}")

    def num_frames(self, num_samples: th.Tensor) -> th.Tensor:
        """
        Compute number of the frames
        """
        if th.sum(num_samples <= self.frame_len):
            raise RuntimeError(
                f"Audio samples less than frame_len ({self.frame_len})")
        num_ffts = self.K.shape[-1]
        if self.center:
            num_samples += num_ffts
        return (num_samples - num_ffts) // self.frame_hop + 1

    def extra_repr(self) -> str:
        return self.expr


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, inverse=False, **kwargs)

    def forward(
            self,
            wav: th.Tensor,
            output: str = "complex"
    ):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        Args
            wav (Tensor) input signal, N x (C) x S
        Return
            transform (Tensor or [Tensor, Tensor]), N x (C) x F x T
        """
        return _forward_stft(wav,
                             self.K,
                             output=output,
                             frame_hop=self.frame_hop,
                             pre_emphasis=self.pre_emphasis,
                             onesided=self.onesided,
                             center=self.center)


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, inverse=True, **kwargs)

    def forward(self,
                transform,
                input: str = "complex") -> th.Tensor:
        """
        Accept phase & magnitude and output raw waveform
        Args
            transform (Tensor or [Tensor, Tensor]), STFT output
        Return
            s (Tensor), N x S
        """
        return _inverse_stft(transform,
                             self.K,
                             self.w,
                             input=input,
                             frame_hop=self.frame_hop,
                             onesided=self.onesided,
                             center=self.center)

    

class MelTransform(nn.Module):
    """
    Perform mel tranform (multiply mel filters)
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        sr: sample rate of souce signal
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        mel_filter: if not "", load mel filter from this
        requires_grad: make it trainable or not
    """

    def __init__(self,
                 frame_len: int,
                 round_pow_of_two: bool = True,
                 sr: int = 16000,
                 num_mels: int = 40,
                 fmin: float = 0.0,
                 fmax: float = None,
                 mel_matrix: str = "",
                 coeff_norm: bool = False,
                 requires_grad: bool = False) -> None:
        super(MelTransform, self).__init__()
        if mel_matrix:
            # pass existed tensor for initialization
            filters = th.load(mel_matrix)
        else:
            # NOTE: the following mel matrix is similiar (not equal to) with
            #       the kaldi results
            filters = mel_filter(frame_len,
                                 round_pow_of_two=round_pow_of_two,
                                 sr=sr,
                                 num_mels=num_mels,
                                 fmax=fmax,
                                 fmin=fmin,
                                 norm=coeff_norm)
        self.num_mels, self.num_bins = filters.shape
        # num_mels x (N // 2 + 1)
        self.filters = nn.Parameter(filters, requires_grad=requires_grad)
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.init = mel_matrix if mel_matrix else "librosa"

    def dim(self) -> int:
        return self.num_mels

    def extra_repr(self) -> str:
        shape = self.filters.shape
        return (f"fmin={self.fmin}, fmax={self.fmax}, " +
                f"mel_filter={shape[0]}x{shape[1]}, init={self.init}")

    def forward(self, linear: th.Tensor) -> th.Tensor:
        """
        Args:
            linear (Tensor): linear spectrogram, N x (C) x T x F
        Return:
            fbank (Tensor): mel-fbank feature, N x (C) x T x B
        """
        if linear.dim() not in [3, 4]:
            raise RuntimeError("MelTransform expect 3/4D tensor, " +
                               f"but got {linear.dim()} instead")
        # N x T x F => N x T x M
        fbank = tf.linear(linear, self.filters.cuda(), bias=None)
        return fbank

class inv_MelTransform(nn.Module):
    """
    Perform mel tranform (multiply mel filters)
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        sr: sample rate of souce signal
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        mel_filter: if not "", load mel filter from this
        requires_grad: make it trainable or not
    """

    def __init__(self,
                 frame_len: int,
                 round_pow_of_two: bool = True,
                 sr: int = 16000,
                 num_mels: int = 40,
                 fmin: float = 0.0,
                 fmax: float = None,
                 mel_matrix: str = "",
                 coeff_norm: bool = False,
                 requires_grad: bool = False) -> None:
        super(inv_MelTransform, self).__init__()
        if mel_matrix:
            # pass existed tensor for initialization
            filters = th.load(mel_matrix)
        else:
            # NOTE: the following mel matrix is similiar (not equal to) with
            #       the kaldi results
            filters = inv_mel_filter(frame_len,
                                     round_pow_of_two=round_pow_of_two,
                                     sr=sr,
                                     num_mels=num_mels,
                                     fmax=fmax,
                                     fmin=fmin,
                                     norm=coeff_norm)
        self.num_mels, self.num_bins = filters.shape
        # num_mels x (N // 2 + 1)
        self.filters = nn.Parameter(filters, requires_grad=requires_grad)
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.init = mel_matrix if mel_matrix else "librosa"

    def dim(self) -> int:
        return self.num_mels

    def extra_repr(self) -> str:
        shape = self.filters.shape
        return (f"fmin={self.fmin}, fmax={self.fmax}, " +
                f"mel_filter={shape[0]}x{shape[1]}, init={self.init}")

    def forward(self, linear: th.Tensor) -> th.Tensor:
        """
        Args:
            linear (Tensor): linear spectrogram, N x (C) x T x F
        Return:
            fbank (Tensor): mel-fbank feature, N x (C) x T x B
        """
        if linear.dim() not in [3, 4]:
            raise RuntimeError("MelTransform expect 3/4D tensor, " +
                               f"but got {linear.dim()} instead")
        # N x T x F => N x T x M
        fbank = tf.linear(linear, self.filters.cuda(), bias=None)
        return fbank

# if __name__ == '__main__':
#     transform = STFT(400, 160)
#     itransform = iSTFT(400, 160)

#     y, fs = sf.read('in.wav')

#     y = th.from_numpy(y)
#     y = y.float()
#     y = y.unsqueeze(0).unsqueeze(0)
#     r,i = transform(y)
#     r,i = r.transpose(2,3),i.transpose(2,3)
#     mag, pha = (r**2+i**2)**0.5, th.atan2(i,r)



#     r, i = mag * th.cos(pha), mag * th.sin(pha)
#     r,i = r.transpose(2,3),i.transpose(2,3)
#     y = itransform((r.squeeze(0),i.squeeze(0)))
#     y = y.squeeze().cpu()
#     y = np.array(y)
#     sf.write('out.wav',y,fs)
