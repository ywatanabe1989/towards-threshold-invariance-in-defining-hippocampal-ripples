#!/usr/bin/env python

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
ap.add_argument(
    "-ftr",
    default="duration",
    choices=["duration", "mep", "ripple peak magnitude"],
    help=" ",
)
args = ap.parse_args()


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np)


################################################################################
## Configures matplotlib
################################################################################
utils.general.configure_mpl(plt)


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST = utils.general.read_txt(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.general.search_str_list(
    LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse
)[1]

lpath_lfp = LPATH_HIPPO_LFP_NPY_LIST_MICE[0]

lpath_rip_sec_GMM = utils.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="GMM_labeled/D{}+".format(args.n_mouse),
)

lpath_rip_sec_CNN = utils.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="CNN_labeled/D{}+".format(args.n_mouse),
)


## Gets Parameters
samp_rate = utils.general.get_samp_rate_int_from_fpath(lpath_lfp)
# dt_sec = 1. / samp_rate


################################################################################
## Loads
################################################################################
lfp = utils.general.load(lpath_lfp)
_rip_sec_GMM = utils.general.load(lpath_rip_sec_GMM)
_rip_sec_CNN = utils.general.load(lpath_rip_sec_CNN)
rip_sec = _rip_sec_GMM
rip_sec["are_ripple_CNN"] = _rip_sec_CNN["are_ripple_CNN"].astype(bool)
del _rip_sec_GMM, _rip_sec_CNN


SD_thres = 7.0
ln_SD_thres = np.log(SD_thres)
are_r_rip_sec = rip_sec["are_ripple_CNN"] & (
    ln_SD_thres < rip_sec["ln(ripple peak magni. / SD)"]
)
r_rip_sec = rip_sec[are_r_rip_sec]  # reasonable ripples
s_rip_sec = rip_sec[~are_r_rip_sec]  # suspicious ripples
