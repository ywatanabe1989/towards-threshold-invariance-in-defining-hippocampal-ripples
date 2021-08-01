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
utils.plt.configure_mpl(plt, legendfontsize="small")


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.general.grep(
    LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse
)[1]

lpath_lfp = LPATH_HIPPO_LFP_NPY_LIST_MICE[0]

lpath_rip_sec_isolated = utils.pj.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="isolated".format(args.n_mouse),
)

lpath_rip_sec_GMM = utils.pj.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="GMM_labeled/D{}+".format(args.n_mouse),
)

lpath_rip_sec_CNN = utils.pj.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="CNN_labeled/D{}+".format(args.n_mouse),
)


## Gets Parameters
samp_rate = utils.pj.get_samp_rate_int_from_fpath(lpath_lfp)
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
GROUP_COLORS = utils.general.load("./conf/global.yaml")["GROUP_COLORS"]
# alpha = 0.5


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


label = "T2T"
ax[0] = fill_color(
    ax[0],
    x,
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
    c=utils.plt.colors.to_RGBA(GROUP_COLORS[label]),
    label=label,
)
label = "F2T"
ax[1] = fill_color(
    ax[1],
    x,
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
    c=utils.plt.colors.to_RGBA(GROUP_COLORS[label]),
    label=label,
)
label = "T2F"
ax[2] = fill_color(
    ax[2],
    x,
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
    c=utils.plt.colors.to_RGBA(GROUP_COLORS[label]),
    label=label,
)
label = "F2F"
ax[3] = fill_color(
    ax[3],
    x,
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
    np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
    c=utils.plt.colors.to_RGBA(GROUP_COLORS[label]),
    label=label,
)
# alpha=alpha,
# label=label,
# facecolor=utils.plt.colors.to_RGBA(),

# ax[0].errorbar(
#     x,
#     np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
#     yerr=np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
#     label=label,
#     color=utils.plt.colors.to_RGBA(GROUP_COLORS[label], alpha=alpha),
# )
# label = "F2T"
# ax[1].errorbar(
#     x,
#     np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
#     yerr=np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
#     label=label,
# )
# label = "T2F"
# ax[2].errorbar(
#     x,
#     np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
#     yerr=np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
#     label=label,
# )
# label = "F2F"
# ax[3].errorbar(
#     x,
#     np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).mean(axis=0),
#     yerr=np.vstack(rip_sec["isolated"][rip_sec[label]]).astype(np.float64).std(axis=0),
#     label=label,
# )
ax[0].set_ylim(-200, 200)
for a in ax:
    a.set_ylabel("Amp. [uV]")
    a.legend(loc="upper left")
fig.suptitle("Average traces triggered by ripple peak magnitude")
ax[3].set_xlabel("Time triggered by ripple peak magnitude [ms]")
ax[3].set_xticks([-200, -100, 0, 100, 200])
# fig.show()
utils.general.save(fig, "average_traces.png")


## EOF
