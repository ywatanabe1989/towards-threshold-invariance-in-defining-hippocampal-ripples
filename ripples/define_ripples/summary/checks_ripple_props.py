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
rip_sec_GMM = utils.general.load(lpath_rip_sec_GMM)
rip_sec_CNN = utils.general.load(lpath_rip_sec_CNN)

rip_sec = rip_sec_GMM
rip_sec["are_ripple_CNN"] = rip_sec_CNN["are_ripple_CNN"].astype(bool)
del rip_sec_GMM, rip_sec_CNN


################################################################################
## Differences among labels
################################################################################
rip_sec["T2T"] = rip_sec["are_ripple_GMM"] & rip_sec["are_ripple_CNN"]
rip_sec["F2T"] = ~rip_sec["are_ripple_GMM"] & rip_sec["are_ripple_CNN"]
rip_sec["T2F"] = rip_sec["are_ripple_GMM"] & ~rip_sec["are_ripple_CNN"]
rip_sec["F2F"] = ~rip_sec["are_ripple_GMM"] & ~rip_sec["are_ripple_CNN"]


################################################################################
## Switches duration/MEP/ripple peak magnitude
################################################################################
if args.ftr == "duration":
    ftr_str = "ln(duration_ms)"
    ylabel = "ln(Duration [ms]) [a.u.]"
    ylim = (2, 8.1)
    n_yticks = 4

if args.ftr == "mep":
    ftr_str = "mean ln(MEP magni. / SD)"
    ylabel = "Mean normalized magnitude of MEP [a.u.]"
    ylim = (-2, 4.1)
    n_yticks = 4

if args.ftr == "ripple peak magnitude":
    ftr_str = "ln(ripple peak magni. / SD)"
    ylabel = "Normalized ripple peak magnitude [a.u.]"
    ylim = (0, 3.1)
    n_yticks = 4


################################################################################
## Plots
################################################################################
utils.general.configure_mpl(plt, figscale=8)
fig, ax = plt.subplots()

colors2str = {
    "T2T": "blue",
    "F2T": "light_blue",
    "T2F": "pink",
    "F2F": "red",
}

dfs = []
alpha = 0.5
ticks = []
for i_label, label in enumerate(["T2T", "F2T", "T2F", "F2F"]):
    df = pd.DataFrame(rip_sec[rip_sec[label]][ftr_str])
    ticks.append("{}\n(n = {:,})".format(label, len(df)))
    RGBA = utils.plt.colors.to_RGBA(colors2str[label], alpha=alpha)
    dfs.append(df)

    box = ax.boxplot(
        x=df,
        boxprops=dict(facecolor=RGBA, color=RGBA),
        medianprops=dict(color="black", linewidth=1),
        notch=False,
        whis=True,
        showfliers=False,
        patch_artist=True,
        positions=[i_label],
    )

ax.set_xticklabels(ticks)
ax.set_ylim(*ylim)

ax.set_ylabel(ylabel)
ax.set_title("Mouse #{}".format(args.n_mouse))

ystart, yend = ax.get_ylim()
ax.yaxis.set_ticks(np.linspace(ystart, np.round(yend, 0), n_yticks))
fig.show()

utils.general.save(
    fig, "mouse_#{}_{}.png".format(args.n_mouse, args.ftr.replace(" ", "_"))
)


## EOF
