#!/usr/bin/env python

import torch
import torch.nn as nn
# from .swish import Swish
# from .Mish.mish import Mish


def init_act_layer(act_str='relu'):
    if act_str == 'relu':
        act_layer = nn.ReLU()
    if act_str == 'lrelu':
        act_layer = nn.LeakyReLU(0.1)
    if act_str == 'mish':
        act_layer = Mish()
    if act_str == 'swish':
        act_layer = Swish()

    return act_layer
          
