#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn

import sys; sys.path.append('.')
from models.ResNet1D.modules import BasicBlock, SecondLevelBlock
from models.modules.act_funcs.init_act_layer import init_act_layer
import julius


class ResNet1D(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()

        self.config = config

        ## Down-sampling Layer
        if self.config['SAMP_RATE_ORIG'] != self.config['samp_rate_tgt']:
            self.resample_layer = julius.resample.ResampleFrac(self.config['SAMP_RATE_ORIG'],
                                                               self.config['samp_rate_tgt'])
        else:
            self.resample_layer = lambda x: x

        
        ## Residual Convolutional Layers
        n_filters = config['n_chs'] * config['n_filters_per_ch']
        
        self.res_conv_blk_layers = nn.Sequential(
              BasicBlock(config['n_chs'], n_filters,
                         activation_str=config['activation_str']),
              *[BasicBlock(n_filters, n_filters, activation_str=config['activation_str'])
                for _ in range(config['n_blks'] - 1)]
             ) # SecondLevel

        ## Global Pooling Layer
        self.global_pooling_layer = \
            self.define_global_pooling_layer(self.config['global_pooling_str'])

        ## FC layer
        n_fc_in = n_filters
        self.fc_layer_diag = nn.Sequential(
            nn.Linear(n_fc_in, self.config['n_fc1']),
            init_act_layer(self.config['activation_str']),
            nn.Dropout(self.config['d_ratio1']),
            nn.Linear(self.config['n_fc1'], self.config['n_fc2']),
            init_act_layer(self.config['activation_str']),
            nn.Dropout(self.config['d_ratio2']),            
            nn.Linear(self.config['n_fc2'], len(self.config['labels'])),
        )
        
        self.fc_layer_subj = nn.Sequential(
            nn.Linear(n_fc_in, self.config['n_fc1']),
            init_act_layer(self.config['activation_str']),
            nn.Dropout(self.config['d_ratio1']),
            nn.Linear(self.config['n_fc1'], self.config['n_fc2']),
            init_act_layer(self.config['activation_str']),
            nn.Dropout(self.config['d_ratio2']),            
            nn.Linear(self.config['n_fc2'], self.config['n_subj_uq_tra']),
        )


    def define_global_pooling_layer(self, global_pooling_str):
        if global_pooling_str== 'average':            
            return nn.AdaptiveAvgPool1d(1)
        if global_pooling_str == 'max':            
            return nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.resample_layer(x)
        x = self.res_conv_blk_layers(x)
        # (7-5-3)*n_blocks => -12*n_blocks => -12*80 => -960
        x = self.global_pooling_layer(x).squeeze(-1) # [16, 304]
        y1 = self.fc_layer_diag(x)
        y2 = self.fc_layer_subj(x)        
        return y1, y2


if __name__ == '__main__':
    # def load_yaml_as_dict(yaml_path = './config.yaml'):
    #     import yaml
    #     config = {}
    #     with open(yaml_path) as f:
    #         _obj = yaml.safe_load(f)
    #         config.update(_obj)
    #     return config
    from utils.general import load_yaml_as_dict
    
    bs, n_chs, seq_len = 16, 19, 1000
    inp = torch.rand(bs, n_chs, seq_len)

    model_config = load_yaml_as_dict('./models/ResNet1D/ResNet1D.yaml')
    model_config['labels'] = ['HV', 'AD', 'DLB', 'iNPH']
    model_config['n_subj_uq_tra'] = 1024
    model_config['n_chs'] = 19
    # model_config['samp_rate_tgt'] = 500
    model = ResNet1D(model_config)

    y1, y2 = model(inp)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(y1[0,0])
    # ax[1].plot(y2[0,0])
    # plt.show()
