#!/usr/bin/env python3

import argparse
import os
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

sys.path.append(".")
import utils
from modules.cliffsDelta.cliffsDelta import cliffsDelta

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
utils.plt.configure_mpl(plt)


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)

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
groups = ["T2T", "F2T", "T2F", "F2F"]
for i_label, label in enumerate(groups):
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
# fig.show()

utils.general.save(
    fig, "mouse_#{}_{}.png".format(args.n_mouse, args.ftr.replace(" ", "_"))
)


################################################################################
## Kruskal-Wallis test (, which is often regarded as a nonparametric version of the ANOVA test.)
################################################################################
data = {g: rip_sec[rip_sec[g]][ftr_str] for g in groups}
H_statistic, p_value = scipy.stats.kruskal(*data.values())  # one-way ANOVA on RANKs
print(
    "\nKruskal-Wallis test:\nH_statistic: {}\np-value: {}\n".format(
        H_statistic, p_value
    )
)


################################################################################
## Brunner-Munzel test (post-hoc test; with Bonferroni correction)
################################################################################
pvals_bm_df = pd.DataFrame(index=groups, columns=groups)

for combi in combinations(groups, 2):
    i_str, j_str = combi
    pvals_bm_df.loc[i_str, j_str] = utils.stats.brunner_munzel_test(
        data[i_str], data[j_str]
    )[1]

NaN_mask = pvals_bm_df.isna()
nonNaN_mask = ~NaN_mask
n_nonNaN = (nonNaN_mask).sum().sum()
## Bonferroni correction
pvals_bm_bonf_df = pvals_bm_df.copy()
pvals_bm_bonf_df[nonNaN_mask] = pvals_bm_bonf_df[nonNaN_mask] * n_nonNaN
## Significant or not
alpha = 0.05
significant_mask = pvals_bm_bonf_df < alpha
significant_mask[NaN_mask] = "-"
print(
    "\nBrunner-Munzel test with Bonferroni correction:\n{}\n".format(significant_mask.T)
)
# out = utils.stats.multicompair(data, groups, testfunc=utils.stats.brunner_munzel_test)

################################################################################
## Cliff's delta values (effect size)
################################################################################
cliff_df = pd.DataFrame(index=groups, columns=groups)

for combi in combinations(groups, 2):
    i_str, j_str = combi
    cliff_df.loc[i_str, j_str] = cliffsDelta(data[i_str], data[j_str])[0]

abs_cliff_df = cliff_df.abs()

## 3d-barplot
# setup the figure and axes
utils.general.configure_mpl(plt, figscale=10)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
## coordinates
_x = np.arange(abs_cliff_df.shape[0])
_y = np.arange(abs_cliff_df.shape[1])
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
dz = [abs_cliff_df.iloc[xi, yi] for xi, yi in zip(x, y)]
dx = dy = 1
z = np.zeros_like(x)
## plot
ax.bar3d(x, y, z, dx, dy, dz, shade=True)
ax.set_zlim(0, 1)
ax.set_xticks(np.arange(len(groups)) + 0.5)
ax.set_xticklabels(groups)
ax.set_yticks(np.arange(len(groups)) + 0.5)
ax.set_yticklabels(groups)
ax.set_title(
    "Absolute cliffs delta values for {ftr_str}\nMouse #{nm}".format(
        ftr_str=ftr_str, nm=args.n_mouse
    )
)
# fig.show()
utils.general.save(fig, "abs_cliffs_delta_mouse_#{}.png".format(args.n_mouse))


## EOF
