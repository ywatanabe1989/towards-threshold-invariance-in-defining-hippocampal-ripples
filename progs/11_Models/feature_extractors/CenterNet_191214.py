import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('./11_Models/')
sys.path.append('./11_Models/feature_extractores/')
from feature_extractors.SincNet import sinc_conv_modified
from feature_extractors.SincNet import SincConv_fast_modified
from feature_extractors.SincNet import SincConv_fast


RESOLUTION_R = 4

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


def center_extract_filtering_k(waveforms, k=1, s=1, p='none'):
    """ Conduct size k kernel's center-extract filtering with padding"""
    if p == 'none':
        p = 0
    if p == 'same':
        p = int(k / 2) + 1
    device = waveforms.device
    bs, n_chs, seq_len = waveforms.shape
    reshaped_waveforms = waveforms.view(-1, seq_len)
    # filters = (torch.ones(k) / k).unsqueeze(0).unsqueeze(0)
    filters = torch.zeros(k)
    filters[int(k/2)+1] = 1
    filters = filters.unsqueeze(0).unsqueeze(0)
    filted = F.conv1d(reshaped_waveforms.unsqueeze(1), filters.to(device), stride=s, padding=p, dilation=1, bias=None, groups=1)
    filted = filted.unsqueeze(dim=1).view(bs, n_chs, seq_len-(k-1)) # fixme
    return filted

    '''
    waveforms = residual
    device = waveforms.device
    k = 7
    s = 1
    p = 0
    # filters = (torch.ones(k) / k).unsqueeze(0).unsqueeze(0).repeat(1, waveforms.shape[1], 1)
    filters = (torch.ones(k) / k).unsqueeze(0).unsqueeze(0).repeat(waveforms.shape[0], waveforms.shape[1], 1)
    F.conv1d(waveforms, filters.to(device), stride=s, padding=p, dilation=1, bias=None, groups=1).shape
    '''


class BasicBlock(nn.Module):
    """ Basic Block using kernel sizes = (7, 5, 3) convolustion with padding"""

    def __init__(self, in_chs, out_chs):
        super(BasicBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.sector7 = build_sector_k(in_chs, out_chs, k=7, p='none')
        self.sector5 = build_sector_k(out_chs, out_chs, k=5, p='none')
        self.sector3 = build_sector_k(out_chs, out_chs, k=3, p='none')

        self.expansion_conv = build_conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = nn.LeakyReLU(0.1)


    def forward(self, x):
        residual = self.bn(x) if self.in_chs == self.out_chs else self.bn(self.expansion_conv(x))

        residual = center_extract_filtering_k(residual, k=7, p='none')
        residual = center_extract_filtering_k(residual, k=5, p='none')
        residual = center_extract_filtering_k(residual, k=3, p='none')

        x = self.sector7(x)
        x = self.sector5(x)
        x = self.sector3(x)

        x += residual
        x = self.activation(x)

        return x


class Filter_402(nn.Module):
    """ kernel sizes = 402 convolustion module aiming to pick a SWR """

    def __init__(self, in_chs, out_chs):
        super(Filter_402, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.sector402 = build_sector_k(in_chs, out_chs, k=402, p='none')

        self.expansion_conv = build_conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = nn.LeakyReLU(0.1)


    def forward(self, x):
        residual = self.bn(x) if self.in_chs == self.out_chs else self.bn(self.expansion_conv(x))

        residual = center_extract_filtering_k(residual, k=402, p='none')

        x = self.sector402(x)

        x += residual
        x = self.activation(x)

        return x



# class Block_31_29_27(nn.Module):
#     """ Basic Block using kernel sizes = (7, 5, 3) convolustion with padding"""

#     def __init__(self, in_chs, out_chs):
#         super(Block_31_29_27, self).__init__()
#         self.in_chs = in_chs
#         self.out_chs = out_chs

#         self.sector31 = build_sector_k(in_chs, out_chs, k=31, p='none')
#         self.sector29 = build_sector_k(out_chs, out_chs, k=29, p='none')
#         self.sector27 = build_sector_k(out_chs, out_chs, k=27, p='none')

#         self.expansion_conv = build_conv_k(in_chs, out_chs, k=1, p=0)

#         self.bn = nn.BatchNorm1d(out_chs)
#         self.activation = nn.LeakyReLU(0.1)


#     def forward(self, x):
#         residual = self.bn(x) if self.in_chs == self.out_chs else self.bn(self.expansion_conv(x))

#         residual = center_extract_filtering_k(residual, k=7, p='none')
#         residual = center_extract_filtering_k(residual, k=5, p='none')
#         residual = center_extract_filtering_k(residual, k=3, p='none')

#         x = self.sector7(x)
#         x = self.sector5(x)
#         x = self.sector3(x)

#         x += residual
#         x = self.activation(x)

#         return x


class CenterNet1D_after_SincNet(nn.Module):
    '''
    After conducting bandpass filtering with SincNet, CenterNet1D predicts heatmap (centers), widths, and offsets
    with ResNet 1D as the backbone.
    The resolusion of the predictions are 1/2 (R=2).

    Input Shape: (Batchsize, 584, 1)
    Output Shape: (Batchsize, 292, 3)
    '''
    def __init__(self, ):
        super(CenterNet1D_after_SincNet, self).__init__()

        # N_filt, Filt_dim, fs = 64, 257, 1000
        N_filt, Filt_dim, fs = 64, 257, 1000
        self.SincNet = SincConv_fast_modified(out_channels=N_filt,
                                              kernel_size=Filt_dim,
                                              sample_rate=fs,
                                              min_low_hz=20,
                                              min_band_hz=50,
                                              )

        self.filter_402 = Filter_402(64, 512)
        '''
        # Mix Bandpass filtered signals with each gain
        self.bands_mix_weights_logits = nn.Parameter((torch.ones(N_filt)).unsqueeze(0).unsqueeze(-1)) # Initialization
        # sigmoid(1) == 0.7311, sigmoid(5) == 0.9933
        self.sigmoid = torch.nn.Sigmoid()
        '''



        self.block1 = BasicBlock(512, 512)
        self.block2 = BasicBlock(512, 512)
        self.block3 = BasicBlock(512, 512)
        self.block4 = BasicBlock(512, 512)
        self.block5 = BasicBlock(512, 512)
        self.block6 = BasicBlock(512, 512)
        self.block7 = BasicBlock(512, 512)
        self.block8 = BasicBlock(512, 512)
        self.block9 = BasicBlock(512, 512)
        self.block10 = BasicBlock(512, 512)

        self.fc = nn.Sequential(nn.Linear(512, 64),
                                nn.LeakyReLU(.1),
                                nn.Dropout(p=.5),
                                nn.Linear(64, 3),
                                )

    def forward(self, x):
        x = x.transpose(1,2)
        dtype = x.dtype

        # Bandpass Filtering
        x = self.SincNet(x)


        '''
        # Mix Bandpass filtered signals with each gain
        bands_mix_weights = self.sigmoid(self.bands_mix_weights_logits)
        x = (x * bands_mix_weights).sum(dim=1, keepdim=True)
        '''
        x = self.filter_402(x)

        # Extract Local Features by 7-5-3 Conv Filters (ResNet Backbone)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        # FC Layer to choose useful channels along time axis
        bs, n_chs, seq_len = x.shape
        x = x.transpose(1,2)
        # x1 = x.clone()
        x = x.reshape(-1, n_chs)
        x = self.fc(x)
        x = x.reshape(bs, seq_len, 3)
        # x4 = x.clone()
        ## Check the validity of the reshaping. Please comment out x1 and x2,
        ## and also turn off Dropout (p=1.) or set the Model to Eval Mode.
        # i = 14
        # j = 3
        # x4[i, j] == self.fc(x1[i, j])

        pred_hm_logits, pred_width_logits, pred_offset_logits = x[..., 0], x[..., 1], x[..., 2]

        pred_hm_prob = pred_hm_logits.sigmoid()
        pred_width = abs(pred_width_logits)
        pred_offset = abs(pred_offset_logits)

        return pred_hm_prob, pred_width, pred_offset




if __name__ == "__main__":
    bs = 16
    seq_len = 1036
    n_features = 1
    data = torch.randn(bs, seq_len, n_features).cuda()

    model = CenterNet1D_after_SincNet()
    model = model.cuda()

    pred_hm_prob, pred_width, pred_offset = model(data) # input: [batch, seq_len, n_features]




    plt.plot(pred_hm_log_prob[0].exp().detach().cpu())
    plt.plot(pred_width[0].detach().cpu())

















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
