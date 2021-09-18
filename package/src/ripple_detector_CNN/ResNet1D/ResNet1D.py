#!/usr/bin/env python3

import sys

import numpy as np
import torch
import torch.nn as nn

from ripple_detector_CNN.ResNet1D.modules import (
    init_act_layer,
    BasicBlock,
    SecondLevelBlock,
)


class ResNet1D(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config

        self.input_bn = nn.BatchNorm1d(config["SEQ_LEN"])

        ## Residual Convolutional Layers
        n_first_filters = config["n_first_filters"]
        self.res_conv_blk_layers = nn.Sequential(
            BasicBlock(
                config["N_CHS"],
                n_first_filters,
                activation_str=config["activation_str"],
            ),
            BasicBlock(
                n_first_filters,
                n_first_filters * 2,
                activation_str=config["activation_str"],
            ),
            BasicBlock(
                n_first_filters * 2,
                n_first_filters * 2,
                activation_str=config["activation_str"],
            ),
        )

        ## Global Pooling Layer
        self.gap_layer = nn.AdaptiveAvgPool1d(1)

        ## FC layer
        n_fc_in = n_first_filters * 2
        self.fc_layer = nn.Sequential(
            nn.Linear(n_fc_in, self.config["n_fc1"]),
            init_act_layer(self.config["activation_str"]),
            nn.Dropout(self.config["d_ratio1"]),
            nn.Linear(self.config["n_fc1"], len(self.config["LABELS"])),
        )

    def forward(self, x):
        if x.ndim != 2:
            x = x.squeeze()  # [BS, SEQ_LEN]
        x = self.input_bn(x)
        x = x.unsqueeze(1)  # [BS, 1, SEQ_LEN] # 1 is N_CHS
        x = self.res_conv_blk_layers(x)
        x = self.gap_layer(x).squeeze(-1)  # [16, 304]
        y = self.fc_layer(x)
        return y


if __name__ == "__main__":
    import mngs

    ## Data
    BS, N_CHS, SEQ_LEN = 16, 1, 400
    inp = torch.rand(BS, N_CHS, SEQ_LEN)

    ## Model
    model_config = mngs.general.load(
        "ripple_detector_CNN/ResNet1D/ResNet1D.yaml"
    )  # fixme
    model = ResNet1D(model_config)

    ## Forward
    y = model(inp)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(y1[0,0])
    # ax[1].plot(y2[0,0])
    # plt.show()
