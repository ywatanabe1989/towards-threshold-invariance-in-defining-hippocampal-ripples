#!/usr/bin/env python3
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
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
args = ap.parse_args()


################################################################################
## Functions
################################################################################
def fill_color(ax, x, mean, std, c, label=None):
    label = "{} (mean +/- std.)".format(label)

    ## mean
    ax.plot(x, mean, color=c, alpha=0.8, label=label)

    ## std
    lower = mean - std
    upper = mean + std
    ax.fill_between(
        x,
        lower,
        upper,
        facecolor=c,
        alpha=0.2,
    )

    # upper edge
    ax.plot(x, upper, linestyle="dotted", color=c, alpha=0.3)
    # lower edge
    ax.plot(x, lower, linestyle="dotted", color=c, alpha=0.3)

    return ax


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np)


################################################################################
## Configures matplotlib
################################################################################
utils.plt.configure_mpl(plt, figsize=(18.1, 12.0), fontsize=8, legendfontsize="small")


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)
lpath_lfp = LPATH_HIPPO_LFP_NPY_LIST_MICE[0]
## Gets Parameters
SAMP_RATE = utils.pj.get_samp_rate_int_from_fpath(lpath_lfp)


################################################################################
## Loads
################################################################################
lfp = utils.general.load(lpath_lfp)  # an electrode; fixme

rip_sec_isolated = utils.pj.load.rip_sec(
    lpath_lfp,
    rip_sec_ver="isolated",
    cycle_dataset=True,
    n_mouse=args.n_mouse,
)

rip_sec_CNN = utils.pj.load.rip_sec(
    lpath_lfp,
    rip_sec_ver="CNN_labeled/D{}-".format(args.n_mouse),
    cycle_dataset=True,
    n_mouse=args.n_mouse,
)


rip_sec = rip_sec_isolated
rip_sec["are_ripple_GMM"] = rip_sec_CNN["are_ripple_GMM"].astype(bool)
rip_sec["are_ripple_CNN"] = rip_sec_CNN["are_ripple_CNN"].astype(bool)
del rip_sec_isolated, rip_sec_CNN


################################################################################
## Packs LFP
################################################################################
extra_sec = 10
rip_sec["LFP"] = [
    lfp[int((row.start_sec) * SAMP_RATE) : int((row.end_sec) * SAMP_RATE)]
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
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
x = np.arange(400) - 200
GROUP_COLORS = utils.general.load("./conf/global.yaml")["GROUP_COLORS"]

for i_row, (label, ax) in enumerate(zip(["T2T", "F2T", "T2F", "F2F"], axes)):
    ax = fill_color(
        ax,
        x,
        np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
        np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
        c=utils.plt.colors.to_RGBA(GROUP_COLORS[label]),
        label=label,
    )
    ax.legend(loc="upper left")

ax[-1].set_xlabel("Time triggered by ripple peak magnitude [ms]")
# a.set_ylabel("Amp. [uV]")
fig.suptitle("Average traces triggered by ripple peak magnitude")
ax[0].set_ylim(-200, 200)
ax[-1].set_xticks([-200, -100, 0, 100, 200])
fig.show()
# utils.general.save(fig, "average_traces.png")


## EOF
