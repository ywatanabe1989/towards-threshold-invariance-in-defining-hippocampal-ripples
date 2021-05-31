#!/usr/bin/env python
import argparse
import os
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(".")
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="02", choices=["01", "02", "03", "04", "05"], help=" "
)  # '01'
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

lpath_rip_sec_isolated = utils.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="isolated".format(args.n_mouse),
)

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
rip_sec_isolated = utils.general.load(lpath_rip_sec_isolated)
rip_sec_GMM = utils.general.load(lpath_rip_sec_GMM)
rip_sec_CNN = utils.general.load(lpath_rip_sec_CNN)

rip_sec = rip_sec_isolated
rip_sec["are_ripple_GMM"] = rip_sec_GMM["are_ripple_GMM"].astype(bool)
rip_sec["are_ripple_CNN"] = rip_sec_CNN["are_ripple_CNN"].astype(bool)
del rip_sec_isolated, rip_sec_GMM, rip_sec_CNN


################################################################################
## Packs LFP
################################################################################
extra_sec = 10
rip_sec["LFP"] = [
    lfp[int((row.start_sec) * samp_rate) : int((row.end_sec) * samp_rate)]
    for i_row, row in rip_sec.iterrows()
]


################################################################################
## Differences among labels
################################################################################
rip_sec["T2T"] = rip_sec["are_ripple_GMM"] & rip_sec["are_ripple_CNN"]
rip_sec["F2T"] = ~rip_sec["are_ripple_GMM"] & rip_sec["are_ripple_CNN"]
rip_sec["T2F"] = rip_sec["are_ripple_GMM"] & ~rip_sec["are_ripple_CNN"]
rip_sec["F2F"] = ~rip_sec["are_ripple_GMM"] & ~rip_sec["are_ripple_CNN"]


################################################################################
## Plots average traces of T2T, F2T, T2F, and  F2F groups
################################################################################
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
x = np.arange(400) - 200
ax[0].errorbar(
    x,
    np.vstack(rip_sec["isolated"][rip_sec["T2T"]]).mean(axis=0),
    yerr=np.vstack(rip_sec["isolated"][rip_sec["T2T"]]).std(axis=0),
)
ax[1].errorbar(
    x,
    np.vstack(rip_sec["isolated"][rip_sec["F2T"]]).mean(axis=0),
    yerr=np.vstack(rip_sec["isolated"][rip_sec["F2T"]]).std(axis=0),
)
ax[2].errorbar(
    x,
    np.vstack(rip_sec["isolated"][rip_sec["T2F"]]).mean(axis=0),
    yerr=np.vstack(rip_sec["isolated"][rip_sec["T2F"]]).std(axis=0),
)
ax[3].errorbar(
    x,
    np.vstack(rip_sec["isolated"][rip_sec["F2F"]]).mean(axis=0),
    yerr=np.vstack(rip_sec["isolated"][rip_sec["F2F"]]).std(axis=0),
)
ax[0].set_ylim(-100, 100)
for a in ax:
    a.set_ylabel("Amp. [uV]")
fig.suptitle("Average traces triggered by ripple peak power")
utils.general.save(fig, "average_traces.png")


## EOF
