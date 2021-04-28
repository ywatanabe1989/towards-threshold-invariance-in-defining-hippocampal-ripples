#!/usr/bin/env python
# import pytorch_lightning as pl
import torch.nn as nn
import math
import torch
# from models.ResNet1D.modules.act_funcs.swish import Swish
# from models.ResNet1Dd.modules.act_funcs.Mish.mish import Mish


def calc_out_len(i, k, s, p, d=1):
  o = (i + 2*p - k - (k-1)*(d-1))/s + 1
  return o


def pad_len_for_the_same_length(i, k, s, d=1):
  p = ((i-1)*s -i + k + (k-1)*(d-1)) / 2
  return p


def conv_k(in_chs, out_chs, k=1, s=1, p=1):
    """ Build size k kernel's convolution layer with padding"""
    return nn.Conv1d(in_chs, out_chs, kernel_size=k, stride=s, padding=p, bias=False)


class BasicBlock(nn.Module):
    """ Basic Block using kernel sizes = (7,5,3) convolustion with padding"""
    expansion = 1

    def __init__(self, in_chs, out_chs, activation_str='relu'):
        super(BasicBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        if activation_str == 'relu':
          activation = nn.ReLU()
        if activation_str == 'lrelu':
          activation = nn.LeakyReLU(0.1)
        if activation_str == 'mish':
          activation = Mish()
        if activation_str == 'swish':
          activation = Swish()

        self.conv7 = conv_k(in_chs, out_chs, k=7, p=3)
        self.bn7 = nn.BatchNorm1d(out_chs)
        self.activation7 = activation

        self.conv5 = conv_k(out_chs, out_chs, k=5, p=2)
        self.bn5 = nn.BatchNorm1d(out_chs)
        self.activation5 = activation

        self.conv3 = conv_k(out_chs, out_chs, k=3, p=1)
        self.bn3 = nn.BatchNorm1d(out_chs)
        self.activation3 = activation

        self.expansion_conv = conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = activation


    def forward(self, x):
        residual = x

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.activation7(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation5(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)

        if self.in_chs != self.out_chs:
          residual = self.expansion_conv(residual)
        residual = self.bn(residual)

        x += residual
        x = self.activation(x)

        return x

class SecondLevelBlock(nn.Module):
    """ Second level block using two basic blocks.
        Two basic blocks are connected with channel wide residual connections.
    """

    def __init__(self, n_filters, activation_str='relu'):
        super().__init__()
        self.n_filters = n_filters
        self.basicblock = BasicBlock(n_filters, n_filters,
                         activation_str=activation_str)

    def forward(self, x):
        residual = x
        x = self.basicblock(x)
        return x

if __name__ == '__main__':
    bs, n_chs, seq_len = 16, 19, 1000
    inp = torch.rand(bs, n_chs, seq_len).cuda()

    n_filters = 4
    slb = SecondLevelBlock(n_chs).cuda()


class SecondLevelBlock(nn.Module):
    """ Second level block using two basic blocks.
        Two basic blocks are connected with channel wide residual connections.
    """

    def __init__(self, n_filters, activation_str='relu'):
        super().__init__()
        self.n_filters = n_filters
        self.basicblock = BasicBlock(n_filters, n_filters,
                         activation_str=activation_str)
        self.fc = nn.Linear(n_filters, 1)
        self.dropout_layer = nn.Dropout(.5)
        # self.avg_pooling_layer = nn.AdaptiveAvgPool1d(1)
          
    def forward(self, x):
        # ch_mean_residual = self.avg_pooling_layer(x.transpose(-2, -1)).transpose(-2, -1)
        ch_weighted_mean_residual = self.fc(x.transpose(-2, -1)).transpose(-2, -1)        
        x = self.basicblock(x)
        x += self.dropout_layer(ch_weighted_mean_residual)
        return x

if __name__ == '__main__':
    bs, n_chs, seq_len = 16, 19, 1000
    inp = torch.rand(bs, n_chs, seq_len).cuda()

    n_filters = 4
    slb = SecondLevelBlock(n_chs).cuda()

    out = slb(inp)
      
