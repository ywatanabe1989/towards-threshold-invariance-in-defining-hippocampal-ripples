#!/usr/bin/env python

import numpy as np
import torch


def torch_to_arr(x):
    is_arr = isinstance(x, (np.ndarray, np.generic))
    if is_arr:  # when x is np.array
        return x
    if torch.is_tensor(x):  # when x is torch.tensor
        return x.detach().cpu().numpy()
