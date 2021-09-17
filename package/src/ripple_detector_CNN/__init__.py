#!/usr/bin/env python3
# Time-stamp: "2021-09-17 01:06:06 (ylab)"

from .RippleDetectorCNN import RippleDetectorCNN
from .ResNet1D.ResNet1D import ResNet1D
from .load_th1 import load_th1
from .define_ripple_candidates import define_ripple_candidates

__copyright__ = "Copyright (C) 2021 Yusuke Watanabe"
__version__ = "0.0.13"
__license__ = "GPL3.0"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/ripple_detector_CNN"

__all__ = ["ResNet1D", "RippleDetectorCNN"]
