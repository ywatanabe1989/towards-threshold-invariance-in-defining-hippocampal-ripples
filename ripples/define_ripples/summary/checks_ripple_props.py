#!/usr/bin/env python3

import argparse
import os
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

sys.path.append(".")
import utils
from modules.cliffsDelta.cliffsDelta import cliffsDelta

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ap.add_argument(
#     "-nm", "--n_mouse", default="02", choices=["01", "02", "03", "04", "05"], help=" "
# )
ap.add_argument(
    "-ftr",
    default="duration",
    choices=["duration", "mep", "ripple peak magnitude"],
    help=" ",
)
args = ap.parse_args()

################################################################################
## Functions
################################################################################
def take_mean_and_std(obj_list, n_round=3):
    arr = np.array(obj_list).astype(float)
    return np.nanmean(arr, axis=0).round(n_round), np.nanstd(arr, axis=0).round(n_round)


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np)


################################################################################
## Configures matplotlib
################################################################################
utils.plt.configure_mpl(
    plt,
    dpi=100,
    figsize=(18.1, 6.0),
    fontsize=7,
    legendfontsize="xx-small",
    hide_spines=True,
    tick_size=0.8,
    tick_width=0.2,
)


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## Switches duration/MEP/ripple peak magnitude
################################################################################
if args.ftr == "duration":
    ftr_str = "ln(duration_ms)"
    # ylabel = "ln(Duration [ms]) [a.u.]"
    ylabel = "Duration [ms]"
    ylim = 0, 630
    yticks = [0, 150, 300, 450, 600]
    n_yticks = 5

if args.ftr == "mep":
    ftr_str = "ln(mean MEP magni. / SD)"
    ylabel = "Mean normalized magnitude of MEP [a.u.]"
    ylim = (-2, 4.1)
    n_yticks = 4
    yticks = np.linspace(ylim[0], np.round(ylim[1], 0), n_yticks)

if args.ftr == "ripple peak magnitude":
    ftr_str = "ln(ripple peak magni. / SD)"
    ylabel = "Normalized ripple peak magnitude [a.u.]"
    ylim = (0, 3.1)
    n_yticks = 4
    yticks = np.linspace(ylim[0], np.round(ylim[1], 0), n_yticks)


fig, axes = plt.subplots(1, 5, sharey=True)
rips_sec_all_mice = []
abs_cliffs = []
for i_mouse, n_mouse_str in enumerate(["01", "02", "03", "04", "05"]):
    print(
        "\n--------------------------------------------------\nn_mouse: {}\n".format(
            n_mouse_str
        )
    )
    ################################################################################
    ## FPATHs
    ################################################################################
    LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(n_mouse_str)
    lpath_lfp = LPATH_HIPPO_LFP_NPY_LIST_MICE[0]

    ################################################################################
    ## Loads
    ################################################################################
    rips_sec = utils.pj.load.lfps_rips_sec(
        LPATH_HIPPO_LFP_NPY_LIST_MICE,
        rip_sec_ver="CNN_labeled/D{}-".format(n_mouse_str),
        cycle_dataset=True,
        n_mouse=n_mouse_str,
    )[1]
    rips_sec = pd.concat(rips_sec)
    rips_sec_all_mice.append(rips_sec)

    ################################################################################
    ## Differences among labels
    ################################################################################
    rips_sec["T2T"] = rips_sec["are_ripple_GMM"] & rips_sec["are_ripple_CNN"]
    rips_sec["F2T"] = ~rips_sec["are_ripple_GMM"] & rips_sec["are_ripple_CNN"]
    rips_sec["T2F"] = rips_sec["are_ripple_GMM"] & ~rips_sec["are_ripple_CNN"]
    rips_sec["F2F"] = ~rips_sec["are_ripple_GMM"] & ~rips_sec["are_ripple_CNN"]

    ################################################################################
    ## Plots
    ################################################################################
    ax = axes[i_mouse]
    ax = utils.plt.ax_set_position(fig, ax, 0.3 * i_mouse - 0.8, 7.0, dragv=True)

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
        df = pd.DataFrame(rips_sec[rips_sec[label]][ftr_str])
        if ftr_str == "ln(duration_ms)":  # to duration [ms]
            df = np.exp(df)
        ticks.append("($n$" + " = {:,}) {}".format(len(df), label))
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
            widths=0.3,
        )

    ax.set_xticklabels(ticks, rotation=90, multialignment="right")
    ax.set_ylim(*ylim)

    ax.set_title("Mouse #{}".format(n_mouse_str))

    # ax.yaxis.set_ticks(yticks)
    # if i_mouse != 0:
    #     ax.yaxis.set_ticks([])

    ################################################################################
    ## Kruskal-Wallis test (, which is often regarded as a nonparametric version of the ANOVA test.)
    ################################################################################
    # rips_sec = pd.concat(rips_sec)
    data = {g: rips_sec.loc[rips_sec[g], ftr_str] for g in groups}

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
        "\nBrunner-Munzel test with Bonferroni correction (alpha = {}):\n{}\n".format(
            alpha, significant_mask.T
        )
    )
    # out = utils.stats.multicompair(data, groups, testfunc=utils.stats.brunner_munzel_test)

    ################################################################################
    ## Cliff's delta values (effect size)
    ################################################################################
    abs_cliff_df = pd.DataFrame(index=groups, columns=groups)

    for combi in combinations(groups, 2):
        i_str, j_str = combi
        # cliff_df.loc[i_str, j_str] = round(cliffsDelta(data[i_str], data[j_str])[0], 3)
        abs_cliff_df.loc[i_str, j_str] = "{:.3f}".format(
            abs(cliffsDelta(data[i_str], data[j_str])[0])
        )

    print("\nAbsolute Cliff's delta values:\n{}".format(abs_cliff_df))
    abs_cliffs.append(abs_cliff_df)

    # 0.147, 0.330, 0.474

axes[0].yaxis.set_ticks(yticks)
fig.axes[0].set_ylabel(ylabel)
# fig.show()
## Saves
utils.general.save(fig, "{}.png".format(args.ftr.replace(" ", "_")))


################################################################################
## Calculates mean and std of absolute Cliff's delta values
################################################################################
abs_cliff_mean, abs_cliff_std = take_mean_and_std(abs_cliffs)
abs_cliff_mean_plus_minus_std = pd.DataFrame(
    data=np.array(
        [
            format(m, ".3f") + " +/- " + format(s, ".3f")
            # format(m, '.3f') + " +/- " + format(s, '.3f')
            for m, s in zip(abs_cliff_mean.flatten(), abs_cliff_std.flatten())
        ]
    ).reshape(abs_cliff_mean.shape),
    index=groups,
    columns=groups,
)
abs_cliff_mean_plus_minus_std = abs_cliff_mean_plus_minus_std.replace(
    "nan +/- nan", "-"
)
print(abs_cliff_mean_plus_minus_std)
## Saves
utils.general.save(
    abs_cliff_mean_plus_minus_std,
    "{}_abs_cliff_mean_plus_minus_std.csv".format(args.ftr),
)


# ## 3d-barplot
# # setup the figure and axes
# # utils.plt.configure_mpl(plt, figscale=10)
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ## coordinates
# _x = np.arange(abs_cliff_df.shape[0])
# _y = np.arange(abs_cliff_df.shape[1])
# _xx, _yy = np.meshgrid(_x, _y)
# x, y = _xx.ravel(), _yy.ravel()
# dz = [abs_cliff_df.iloc[xi, yi] for xi, yi in zip(x, y)]
# dx = dy = 1
# z = np.zeros_like(x)
# ## plot
# ax.bar3d(x, y, z, dx, dy, dz, shade=True)
# ax.set_zlim(0, 1)
# ax.set_xticks(np.arange(len(groups)) + 0.5)
# ax.set_xticklabels(groups)
# ax.set_yticks(np.arange(len(groups)) + 0.5)
# ax.set_yticklabels(groups)
# ax.set_title(
#     "Absolute cliffs delta values for {ftr_str}\nMouse #{nm}".format(
#         ftr_str=ftr_str, nm=n_mouse_str
#     )
# )
# fig.show()
# # utils.general.save(fig, "abs_cliffs_delta_mouse_#{}.png".format(n_mouse_str))


# # # ## EOF
