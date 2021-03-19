import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
import torch


def conv_k(in_planes, out_planes, k, stride=1):
    """ Build size k kernel's convolution layer with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=k, stride=stride, padding=1, bias=False)

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

# def conv5x5(in_planes, out_planes, stride=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
#                      padding=1, bias=False)

# def conv7x7(in_planes, out_planes, stride=1):
#     return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
#                      padding=1, bias=False)


class BasicBlock_k(nn.Module):
    """ Basic Block using kernel size = k convolustion with padding"""
    expansion = 1

    def __init__(self, inplanes, planes, k, stride=1, downsample=None):
        super(BasicBlock_k, self).__init__()
        self.conv1 = conv_k(inplanes, planes, k, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_k(planes, planes, k)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class BasicBlock3x3(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes3, planes, stride=1, downsample=None):
#         super(BasicBlock3x3, self).__init__()
#         self.conv1 = conv3x3(inplanes3, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class BasicBlock5x5(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes5, planes, stride=1, downsample=None):
#         super(BasicBlock5x5, self).__init__()
#         self.conv1 = conv5x5(inplanes5, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv5x5(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         d = residual.shape[2] - out.shape[2]
#         out1 = residual[:,:,0:-d] + out
#         out1 = self.relu(out1)
#         # out += residual

#         return out1



# class BasicBlock7x7(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes7, planes, stride=1, downsample=None):
#         super(BasicBlock7x7, self).__init__()
#         self.conv1 = conv7x7(inplanes7, planes, stride)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv7x7(planes, planes)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         d = residual.shape[2] - out.shape[2]
#         out1 = residual[:, :, 0:-d] + out
#         out1 = self.relu(out1)
#         # out += residual

#         return out1




class MultiScaleResNet1D(nn.Module):
    def __init__(self, input_channel, max_seq_len, layers=[1, 1, 1]):
        self.outsize = int((max_seq_len/2)*3)
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MultiScaleResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = x0.transpose(1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]

        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)

        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        y = self.maxpool5(y)

        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool7(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.squeeze()
        # out = self.drop(out)

        return out


# bs = 64
# seq_len = 512
# n_features = 1
# data = torch.randn(bs, seq_len, n_features).cuda().half()
# model = MultiScaleResNet1D(n_features, max_seq_len=seq_len)
# model = model.cuda().half()
# output = model(data) # input: [batch, seq_len, n_features]
# print(output.shape)



bs = 64
seq_len = 512
n_features = 1
data = torch.randn(bs, seq_len, n_features)
# conv = conv_k(in_planes, out_planes, k, stride=1)
