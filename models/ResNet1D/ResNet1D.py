#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn

import sys; sys.path.append('.')
from models.ResNet1D.modules import BasicBlock, SecondLevelBlock
from models.modules.act_funcs.init_act_layer import init_act_layer


class ResNet1D(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()

        self.config = config

        
        self.input_bn = nn.BatchNorm1d(config['seq_len']) # fixme
        
        ## Residual Convolutional Layers
        n_filters = 64
        
        self.res_conv_blk_layers = nn.Sequential(
              BasicBlock(config['n_chs'], n_filters,
                         activation_str=config['activation_str']),
              BasicBlock(n_filters, n_filters*2,
                         activation_str=config['activation_str']),
              BasicBlock(n_filters*2, n_filters*2,
                         activation_str=config['activation_str']),
             ) # SecondLevel


        ## Global Pooling Layer
        self.gap_layer = nn.AdaptiveAvgPool1d(1)

        ## FC layer
        n_fc_in = n_filters*2
        self.fc_layer = nn.Sequential(
            nn.Linear(n_fc_in, self.config['n_fc1']),
            init_act_layer(self.config['activation_str']),
            nn.Dropout(self.config['d_ratio1']),
            nn.Linear(self.config['n_fc1'], self.config['n_fc2']),
            init_act_layer(self.config['activation_str']),
            nn.Dropout(self.config['d_ratio2']),            
            nn.Linear(self.config['n_fc2'], len(self.config['labels'])),
        )
        

    def znorm(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        stds = x.std(dim=-1, keepdims=True)
        means = x.mean(dim=-1, keepdims=True)
        x = (x - means) / stds
        return x.to(dtype)
        
    def forward(self, x):
        x = self.znorm(x)
        x = self.res_conv_blk_layers(x)
        # (7-5-3)*n_blocks => -12*n_blocks => -12*80 => -960
        x = self.gap_layer(x).squeeze(-1) # [16, 304]
        y = self.fc_layer(x)
        return y


if __name__ == '__main__':
    from utils.general import load_yaml_as_dict
    
    bs, n_chs, seq_len = 16, 1, 400
    inp = torch.rand(bs, n_chs, seq_len)

    model_config = load_yaml_as_dict('./models/ResNet1D/ResNet1D.yaml')
    model_config['labels'] = ['nonRipple', 'Ripple']
    model_config['n_chs'] = 1
    model_config['seq_len'] = seq_len
    
    model = ResNet1D(model_config)

    y = model(inp)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(y1[0,0])
    # ax[1].plot(y2[0,0])
    # plt.show()
