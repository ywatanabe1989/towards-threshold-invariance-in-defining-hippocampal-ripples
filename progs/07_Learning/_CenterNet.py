import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
import torch.utils.data as utils

## my own packages
import sys
sys.path.append('./')
# sys.path.append('./utils')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_yolo_191201 import DataloaderFulfiller
sys.path.append('./07_Learning/')
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./11_Models/')
sys.path.append('./11_Models/yolo')
sys.path.append('./11_Models/yolo/utils')
from yolo.models import Darknet
from yolo.data_parallel import DataParallel
from utils.utils import non_max_suppression_1D as nms
from utils.utils import check_samples_1D, plot_prediction_1D
import math


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    def __init__(self,
                 out_channels=50, # n_filters
                 kernel_size=128,
                 sample_rate=1000,
                 in_channels=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 min_low_hz=1,
                 min_band_hz=1):

        super(SincConv_fast, self).__init__()

        assert in_channels == 1

        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1 # Force to be odd
        self.stride = stride
        self.padding = int(kernel_size/2) if padding is None else padding
        self.dilation = dilation

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Hz space
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        hz = np.linspace(min_low_hz, high_hz, self.out_channels + 1)

        self.low_hz = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2*math.pi*n_lin/self.kernel_size);

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_
        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.

        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])

        band_pass = torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        band_pass = band_pass / (2*band[:,None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None, groups=1)




def calc_out_len(i, k, s, p, d=1):
  o = (i + 2*p - k - (k-1)*(d-1))/s + 1
  return o


def pad_len_for_the_same_length(i, k, s, d=1):
  p = ((i-1)*s -i + k + (k-1)*(d-1)) / 2
  return p


def build_conv_k(in_chs, out_chs, k=1, s=1, p=1):
    """ Build size k kernel's convolution layer with padding"""
    return nn.Conv1d(in_chs, out_chs, kernel_size=k, stride=s, padding=p, bias=False)


def build_sector_k(in_chs, out_chs, k=1, p='none'):
    if p == 'none':
        p = 0
    if p == 'same':
        p = math.floor(k / 2)

    convk = build_conv_k(in_chs, out_chs, k=k, p=p)
    bn = nn.BatchNorm1d(out_chs)
    activation = nn.LeakyReLU(0.1)

    sectork = torch.nn.Sequential(
      convk,
      bn,
      activation,
        )

    return sectork


class BasicBlock(nn.Module):
    """ Basic Block using kernel sizes = (7, 5, 3) convolustion with padding"""

    def __init__(self, in_chs, out_chs):
        super(BasicBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.sector7 = build_sector_k(in_chs, out_chs, k=7, p='same')
        self.sector5 = build_sector_k(out_chs, out_chs, k=5, p='same')
        self.sector3 = build_sector_k(out_chs, out_chs, k=3, p='same')

        self.expansion_conv = build_conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = nn.LeakyReLU(0.1)


    def forward(self, x):
        residual = x

        x = self.sector7(x)
        x = self.sector5(x)
        x = self.sector3(x)

        if self.in_chs != self.out_chs:
            residual = self.expansion_conv(residual)
        residual = self.bn(residual)

        x += residual
        x = self.activation(x)

        return x


class MiddleBlock(nn.Module):
    """ Middle Block using kernel sizes = (29, 17, 3) convolustion with padding"""

    def __init__(self, in_chs, out_chs):
        super(MiddleBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.sector29 = build_sector_k(in_chs, out_chs, k=29)
        self.sector5 = build_sector_k(out_chs, out_chs, k=5)
        self.sector3 = build_sector_k(out_chs, out_chs, k=3)

        self.expansion_conv = build_conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = nn.LeakyReLU(0.1)


    def forward(self, x):
        residual = x
        if self.in_chs != self.out_chs:
            residual = self.expansion_conv(residual)
        residual = self.bn(residual)

        x = self.sector29(x)
        x = self.sector17(x)
        x = self.sector3(x)

        x += residual
        x = self.activation(x)

        return x



class BiggerBlock(nn.Module):
    """ Bigger using kernel sizes = odd numbers ranging from 31 to 3 convolustion with padding"""
    def __init__(self, in_chs, out_chs):
        super(BiggerBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.expansion_conv = build_conv_k(in_chs, out_chs, k=1, p=0)
        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = nn.LeakyReLU(0.1)

        self.sector31 = build_sector_k(in_chs, out_chs, k=31)
        self.sector29 = build_sector_k(out_chs, out_chs, k=29)
        self.sector27 = build_sector_k(out_chs, out_chs, k=27)

        self.sector25 = build_sector_k(out_chs, out_chs, k=25)
        self.sector23 = build_sector_k(out_chs, out_chs, k=23)
        self.sector21 = build_sector_k(out_chs, out_chs, k=21)

        self.sector19 = build_sector_k(out_chs, out_chs, k=19)
        self.sector17 = build_sector_k(out_chs, out_chs, k=17)
        self.sector15 = build_sector_k(out_chs, out_chs, k=15)

        self.sector13 = build_sector_k(out_chs, out_chs, k=13)
        self.sector11 = build_sector_k(out_chs, out_chs, k=11)
        self.sector9 = build_sector_k(out_chs, out_chs, k=9)

        self.sector7 = build_sector_k(out_chs, out_chs, k=7)
        self.sector5 = build_sector_k(out_chs, out_chs, k=5)
        self.sector3 = build_sector_k(out_chs, out_chs, k=3)

    def forward_residual_block(self, x, sector_L, sector_M, sector_S, expansion=False):
        residual = self.bn(self.expansion_conv(x)) if expansion is True else self.bn(x)
        x = sector_L(x)
        x = sector_M(x)
        x = sector_S(x)
        x += residual
        x = self.activation(x)
        return x

    def forward(self, x):
        expansion = True if self.in_chs != self.out_chs else False
        x = self.forward_residual_block(x, self.sector31, self.sector29, self.sector27, expansion=expansion)
        x = self.forward_residual_block(x, self.sector25, self.sector23, self.sector21, expansion=False)
        x = self.forward_residual_block(x, self.sector19, self.sector17, self.sector15, expansion=False)
        x = self.forward_residual_block(x, self.sector13, self.sector11, self.sector9, expansion=False)
        x = self.forward_residual_block(x, self.sector7, self.sector5, self.sector3, expansion=False)
        return x



class ResNet1D(nn.Module): # Original
    def __init__(self, ):
        super(ResNet1D, self).__init__()

        self.block1 = BasicBlock(1, 64)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 128)

        self.gap = torch.mean

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        dtype_orig = x.dtype
        x = x.to(torch.float64)
        x = self.gap(x, dim=-1)
        x = x.to(dtype_orig)

        x = self.activation(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


# class CenterNet1D(nn.Module): # Orig, Maybe, the kernel sizes are too small.
#     def __init__(self, ):
#         super(CenterNet1D, self).__init__()

#         # self.SincNet = SincConv_fast(out_channels=8)

#         self.block1 = BasicBlock(1, 64)
#         self.block2 = BasicBlock(64, 128)
#         self.block3 = BasicBlock(128, 256)

#         # self.log_sigmoid = nn.LogSigmoid()


#     def forward(self, x):
#         x = x.transpose(1,2)
#         dtype = x.dtype

#         # x = self.SincNet(x)

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         dtype_orig = x.dtype
#         x = x.to(torch.float64)

#         # Global Average Pooling
#         pred_hm_logits, _ = x[:, int(x.shape[1]/2):, :].max(dim=1)
#         pred_width_logits = (x[:, :int(x.shape[1]/2), :].mean(dim=1))

#         pred_width = abs(pred_width_logits)

#         return pred_hm_logits.to(dtype), pred_width.to(dtype)


class CenterNet1D(nn.Module):
    def __init__(self, ):
        super(CenterNet1D, self).__init__()

        self.ripple_bandpass = BandPass(lo_hz=150, hi_hz=250, kernel_size=101, padding='same')

        self.block1 = BasicBlock(1, 64)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 128)
        self.block4 = BasicBlock(128, 128)
        self.block5 = BasicBlock(128, 128)

    def forward(self, x):
        x = x.transpose(1,2)
        dtype = x.dtype

        x = self.ripple_bandpass(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        dtype_orig = x.dtype
        x = x.to(torch.float64)

        # Global Average Pooling
        pred_hm_logits = x[:, :int(x.shape[1]/2), :].mean(dim=1) # fixme
        pred_width_logits = (x[:, int(x.shape[1]/2):, :].mean(dim=1))

        pred_hm_prob = pred_hm_logits.sigmoid()
        pred_width = abs(pred_width_logits)

        return pred_hm_prob.to(dtype), pred_width.to(dtype)



class BandPass(nn.Module):
    '''
    Conduct Bandpass Filtering in mini-batch manner with GPU if input 1D signal is on GPU.

    This class has no parameters to learn; however, to explicitly print out with model's layers on ipython, in my case,
    I made this BandPass class to inherit the torch.nn.Module class.

    Input Shape: (Batchsize, 1, SignalLength)
    Output Shape: (Batchsize, 1, SignalLength - (kernel_size-1)) if padding == 'none' (default)
    '''

    def __init__(self,
                 lo_hz=150,
                 hi_hz=250,
                 sample_rate=1000,
                 kernel_size=101,
                 stride=1,
                 padding='none',
                 dilation=1,
    ):

        super(BandPass, self).__init__()

        # Force kernel_size to be odd
        if kernel_size % 2 != 0:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = kernel_size + 1
            print('Kernel Size must be odd. It was modified to {}'.format(kernel_size))

        self.stride = stride

        if padding == 'none': # default
            self.padding = 0
        if padding == 'same':
            self.padding = int(kernel_size / 2)

        self.dilation = dilation

        self.filters = self._create_ripple_bandpass_filter(fs=sample_rate,
                                                           lo_hz=lo_hz,
                                                           hi_hz=hi_hz,
                                                           order=kernel_size)
        self.filters = torch.FloatTensor(self.filters).unsqueeze(0).unsqueeze(0)

    def _create_ripple_bandpass_filter(self, fs=1000, lo_hz=150, hi_hz=250, order=101):
        '''
        As used in Kay et al., 2016.
        '''
        from scipy.signal import remez
        ORDER = order # ORDER = 101
        nyquist = 0.5 * fs
        TRANSITION_BAND = 25
        RIPPLE_BAND = [lo_hz, hi_hz]
        desired = [0,
                   RIPPLE_BAND[0] - TRANSITION_BAND, RIPPLE_BAND[0],
                   RIPPLE_BAND[1],
                   RIPPLE_BAND[1] + TRANSITION_BAND,
                   nyquist]
        return remez(ORDER, desired, [0, 1, 0], Hz=fs)

    def __call__(self, waveforms):
        device = waveforms.device
        return F.conv1d(waveforms,
                        self.filters.to(device),
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


if __name__ == "__main__":
    bs = 2
    seq_len = 512
    n_features = 1

    data = torch.randn(bs, seq_len, n_features).cuda()
    model = BandPass(lo_hz=150, hi_hz=250, kernel_size=101)

    plt.plot(model.filters.squeeze())

    out = model(data.transpose(1,2))



    model = SincConv_fast(out_channels=1).cuda()

    plt.plot(model.filters.squeeze())

    out = model(data.transpose(1,2))















    bs = 2
    seq_len = 512
    n_features = 1

    data = torch.randn(bs, seq_len, n_features).cuda()
    model = CenterNet1D()
    model = model.cuda()

    pred_hm_log_prob, pred_width = model(data) # input: [batch, seq_len, n_features]

    plt.plot(pred_hm_log_prob[0].exp().detach().cpu())
    plt.plot(pred_width[0].detach().cpu())
















    data = torch.randn(bs, seq_len, n_features).cuda()
    model = ResNet1D()
    model = model.cuda()

    output = model(data) # input: [batch, seq_len, n_features]
    print(output.shape)


    n_filters = 10
    kernel_size = 100
    padding = int(kernel_size/2)
    model = SincConv_fast()
    model = model.cuda()
    output = model(data.transpose(1,2)) # input: [batch, seq_len, n_features]

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
