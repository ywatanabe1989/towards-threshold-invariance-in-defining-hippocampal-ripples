import torch.nn as nn
import math
import torch

def calc_out_len(i, k, s, p, d=1):
  o = (i + 2*p - k - (k-1)*(d-1))/s + 1
  return o

def pad_len_for_the_same_length(i, k, s, d=1):
  p = ((i-1)*s -i + k + (k-1)*(d-1)) / 2
  return p

def conv_k(in_chs, out_chs, k=1, s=1, p=1):
    """ Build size k kernel's convolution layer with padding"""
    return nn.Conv1d(in_chs, out_chs, kernel_size=k, stride=s, padding=p, bias=False)

# def test(seq_len, k, s=1, d=1):
#   bs = 64
#   # seq_len = 1024
#   n_features = 1
#   data = torch.randn(bs, n_features, seq_len)
#   # k = 11
#   # s = 1
#   # d = 1
#   # if (k % 2) == 0:
#   #   k += 1
#   p = int(pad_len_for_the_same_length(seq_len, k, s, d))
#   conv = conv_k(1, 32, k, s=s, p=p)
#   out = conv(data) # 64, 32, 512)
#   calced_outlen = int(calc_out_len(seq_len, k, s, p))
#   if out.shape[-1] == seq_len:
#       print("Same")
#   if out.shape[-1] == calced_outlen:
#       print('Calculation is correct')
#   return p, calced_outlen


class BasicBlock(nn.Module):
    """ Basic Block using kernel sizes = (7,5,3) convolustion with padding"""
    expansion = 1

    def __init__(self, in_chs, out_chs, use_activation):
        super(BasicBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs

        if use_activation == 'lrelu':
          activation = nn.LeakyReLU(0.1)
        if use_activation == 'relu':
          activation = nn.ReLU()
        if use_activation == 'selu':
          activation = nn.SELU()

        self.conv7 = conv_k(in_chs, out_chs, k=7, p=3)
        self.bn7 = nn.BatchNorm1d(out_chs)
        self.activation7 = activation        # self.selu7 = nn.SELU(inplace=True) # self.relu7 = nn.ReLU()

        self.conv5 = conv_k(out_chs, out_chs, k=5, p=2)
        self.bn5 = nn.BatchNorm1d(out_chs)
        self.activation5 = activation        # self.selu5 = nn.SELU(inplace=True) # self.relu5 = nn.ReLU()

        self.conv3 = conv_k(out_chs, out_chs, k=3, p=1)
        self.bn3 = nn.BatchNorm1d(out_chs)
        self.activation3 = activation        # self.selu3 = nn.SELU(inplace=True) # self.relu3 = nn.ReLU()

        self.expansion_conv = conv_k(in_chs, out_chs, k=1, p=0)

        self.bn = nn.BatchNorm1d(out_chs)
        self.activation = activation        # self.selu = nn.SELU(inplace=True) # self.relu = nn.ReLU()


    def forward(self, x):
        residual = x

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.activation7(x)        # x = self.selu7(x) # self.relu7(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation5(x)        # x = self.selu5(x) # self.relu5(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation3(x)        # x = self.selu3(x) # self.relu3(x)

        if self.in_chs != self.out_chs:
          residual = self.expansion_conv(residual)
        residual = self.bn(residual)

        x += residual
        x = self.activation(x)        # x = self.selu(x) # self.relu(x)

        return x


# class ResNet1D(nn.Module): # original
#     def __init__(self, in_chs1): # seq_len
#         super(ResNet1D, self).__init__()

#         self.in_chs1 = in_chs1
#         self.out_chs1 = 64

#         self.in_chs2 = self.out_chs1
#         self.out_chs2 = 128

#         self.in_chs3 = self.out_chs2
#         self.out_chs3 = 128

#         self.outsize = self.out_chs3

#         self.block1 = BasicBlock(self.in_chs1, self.out_chs1)
#         self.block2 = BasicBlock(self.in_chs2, self.out_chs2)
#         self.block3 = BasicBlock(self.in_chs3, self.out_chs3)

#         self.gap = torch.mean # nn.AvgPool1d(kernel_size=16, stride=1, padding=0) # fixme

#     def forward(self, x):
#         x = x.transpose(1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         dtype_orig = x.dtype
#         x = x.to(torch.float64)
#         x = self.gap(x, dim=-1)
#         x = x.to(dtype_orig)

#         return x

# class ResNet1D(nn.Module): # Original, working, 2019/12/28
#     def __init__(self, in_chs1, out_ch1, use_activation): # seq_len
#         super(ResNet1D, self).__init__()

#         self.in_chs1 = in_chs1
#         self.out_chs1 = out_ch1 # 64

#         self.in_chs2 = self.out_chs1
#         self.out_chs2 = self.out_chs1*2 # 128

#         self.in_chs3 = self.out_chs2
#         self.out_chs3 = self.out_chs2 # 128

#         self.outsize = self.out_chs3

#         self.block1 = BasicBlock(self.in_chs1, self.out_chs1, use_activation)
#         self.block2 = BasicBlock(self.in_chs2, self.out_chs2, use_activation)
#         self.block3 = BasicBlock(self.in_chs3, self.out_chs3, use_activation)

#         self.gap = torch.mean # nn.AvgPool1d(kernel_size=16, stride=1, padding=0) # fixme

#     def forward(self, x):
#         x = x.transpose(1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         dtype_orig = x.dtype
#         x = x.to(torch.float64)
#         x = self.gap(x, dim=-1)
#         x = x.to(dtype_orig)

#         return x

class ResNet1D(nn.Module): # Original
    def __init__(self, in_chs1, out_ch1, use_activation): # seq_len
        super(ResNet1D, self).__init__()

        self.in_chs1 = in_chs1
        self.out_chs1 = out_ch1 # 64

        self.in_chs2 = self.out_chs1
        self.out_chs2 = self.out_chs1*2 # 128

        self.in_chs3 = self.out_chs2
        self.out_chs3 = self.out_chs2 # 128

        self.outsize = self.out_chs3

        self.block1 = BasicBlock(self.in_chs1, self.out_chs1, use_activation)
        self.block2 = BasicBlock(self.in_chs2, self.out_chs2, use_activation)
        self.block3 = BasicBlock(self.in_chs3, self.out_chs3, use_activation)

        self.gap = torch.mean # nn.AvgPool1d(kernel_size=16, stride=1, padding=0) # fixme

    def forward(self, x):
        x = x.transpose(1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        dtype_orig = x.dtype

        x = x.to(torch.float32)
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True)
        x = (x-m)/s # Normalization
        x = x.to(dtype_orig)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)


        x = x.to(torch.float64)
        x = self.gap(x, dim=-1)
        x = x.to(dtype_orig)

        return x



# class ResNet1D(nn.Module): # split to half before Averaging
#     def __init__(self, in_chs1, out_ch1, use_activation): # seq_len
#         super(ResNet1D, self).__init__()

#         self.in_chs1 = in_chs1
#         self.out_chs1 = out_ch1 # 64

#         self.in_chs2 = self.out_chs1
#         self.out_chs2 = self.out_chs1*2 # 128

#         self.in_chs3 = self.out_chs2
#         self.out_chs3 = self.out_chs2 # 128

#         self.outsize = self.out_chs3

#         self.block1 = BasicBlock(self.in_chs1, self.out_chs1, use_activation)
#         self.block2 = BasicBlock(self.in_chs2, self.out_chs2, use_activation)
#         self.block3 = BasicBlock(self.in_chs3, self.out_chs3, use_activation)

#         self.ap = torch.mean # nn.AvgPool1d(kernel_size=16, stride=1, padding=0) # fixme
#         self.mp = torch.max

#     def forward(self, x):
#         x = x.transpose(1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         dtype_orig = x.dtype
#         x = x.to(torch.float64)

#         x1 = self.ap(x[:, :, int(x.shape[-1]/2):], dim=-1).unsqueeze(-1)
#         x2 = self.ap(x[:, :, :int(x.shape[-1]/2)], dim=-1).unsqueeze(-1)

#         x = torch.cat((x1, x2), dim=-1)

#         x = self.mp(x, dim=-1)[0]

#         x = x.to(dtype_orig)

#         return x





'''
bs = 64
seq_len = 2048
n_features = 1

data = torch.randn(bs, seq_len, n_features).cuda()
model = ResNet1D(n_features)
model = model.cuda()

output = model(data) # input: [batch, seq_len, n_features]
print(output.shape)
'''
