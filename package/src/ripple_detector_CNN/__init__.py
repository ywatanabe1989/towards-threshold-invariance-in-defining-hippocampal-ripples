#!/usr/bin/env python3
# Time-stamp: "2021-09-18 07:41:49 (ywatanabe)"

from .RippleDetectorCNN import RippleDetectorCNN
from .ResNet1D.ResNet1D import ResNet1D
from .define_ripple_candidates import define_ripple_candidates

__copyright__ = "Copyright (C) 2021 Yusuke Watanabe"
__version__ = "0.1.3"
__license__ = "GPL3.0"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/ripple_detector_CNN"

__all__ = ["ResNet1D", "RippleDetectorCNN"]
