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
from matplotlib import colors

sys.path.append(".")
import utils
from modules.cliffsDelta.cliffsDelta import cliffsDelta

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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


def take_median_and_std(obj_list, n_round=3):
    arr = np.array(obj_list).astype(float)
    return np.nanmedian(arr, axis=0).round(n_round), np.nanstd(arr, axis=0).round(
        n_round
    )


def judge_cliff(cliff_val):
    abs_cliff_val = abs(cliff_val)
    if 1.0 < abs_cliff_val:
        return "Error; Cliff's delta value must be [-1, 1]"
    if 0.474 <= abs_cliff_val:
        return "large"
    if 0.330 <= abs_cliff_val:
        return "medium"
    if 0.147 <= abs_cliff_val:
        return "small"
    if 0.0 <= abs_cliff_val:
        return "negligible"


def to_med_mad_df(listed_scalars):
    df = pd.DataFrame(listed_scalars)
    med = pd.DataFrame(df.median()).T
    mad = pd.DataFrame(data=scipy.stats.median_abs_deviation(df), index=med.columns).T

    out_df = med.append(mad)
    out_df = out_df.append(df)

    index = ["median", "median absolute deviation"] + list(
        np.arange(len(df)).astype(int)
    )

    out_df.index = index

    return out_df


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
    ylabel = "Duration [ms]"
    ylim = 0, 630
    yticks = [0, 150, 300, 450, 600]
    n_yticks = 5

if args.ftr == "mep":
    ftr_str = "ln(mean MEP magni. / SD)"
    ylabel = "ln(Mean normalized \nmagnitude of MEP) [a.u.]"
    # ylim = (-2, 4.1)
    ylim = (-2, 4.1)
    n_yticks = 4
    yticks = np.linspace(ylim[0], np.round(ylim[1], 0), n_yticks)

if args.ftr == "ripple peak magnitude":
    ftr_str = "ln(ripple peak magni. / SD)"
    ylabel = "ln(Normalized ripple \npeak magnitude) [a.u.]"
    ylim = (0, 3.1)
    n_yticks = 4
    yticks = np.linspace(ylim[0], np.round(ylim[1], 0), n_yticks)


n_mice_str = ["01", "02", "03", "04", "05"]
fig, axes = plt.subplots(1, len(n_mice_str))
rips_sec_all_mice = []
abs_cliffs = []
groups = ["T2T", "F2T", "T2F", "F2F"]
meds_dict = utils.general.listed_dict(groups)
for i_mouse, n_mouse_str in enumerate(n_mice_str):
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
    # ax = utils.plt.ax_set_position(fig, ax, 0.1 * i_mouse - 0.8, 7.0, dragv=True)
    ax = utils.plt.ax_set_position(fig, ax, 0.0, 7.0, dragv=True)

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

        meds_dict[label].append(np.median(df))

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
        abs_cliff_df.loc[i_str, j_str] = "{:.3f}".format(
            abs(cliffsDelta(data[i_str], data[j_str])[0])
        )

    print("\nAbsolute Cliff's delta values:\n{}".format(abs_cliff_df))
    abs_cliffs.append(abs_cliff_df)

    ax.yaxis.set_ticks(yticks)
    ax.set_yticklabels([])

# axes[0].yaxis.set_ticks(yticks)
axes[0].set_yticklabels(yticks)
fig.axes[0].set_ylabel(ylabel)
# fig.show()
## Saves
spath = utils.general.mk_spath(
    "distribution/{}.tiff".format(args.ftr.replace(" ", "_"))
)
utils.general.save(fig, spath)
# utils.general.save(fig, "{}_distribution.tiff".format(args.ftr.replace(" ", "_")))


################################################################################
## Calculates mean and std of absolute Cliff's delta values
################################################################################
abs_cliff_mean, abs_cliff_std = take_mean_and_std(abs_cliffs)
judge_arr = (
    np.array(["-" for _ in range(abs_cliff_mean.size)])
    .reshape(abs_cliff_mean.shape)
    .astype(object)
)

for i in range(abs_cliff_mean.shape[0]):
    for j in range(abs_cliff_mean.shape[1]):
        if not np.isnan(abs_cliff_mean[i, j]):
            judge_arr[i, j] = judge_cliff(abs_cliff_mean[i, j])


abs_cliff_mean_plus_minus_std = pd.DataFrame(
    data=np.array(
        [
            format(m, ".3f") + " +/- " + format(s, ".3f")
            for m, s in zip(abs_cliff_mean.flatten(), abs_cliff_std.flatten())
        ]
    ).reshape(abs_cliff_mean.shape),
    index=groups,
    columns=groups,
).replace("nan +/- nan", "-")

# abs_cliff_mean_plus_minus_std = abs_cliff_mean_plus_minus_std.replace("nan +/- nan", "-")

## Adds judges for the mean cliff's delta values
abs_cliff_mean_plus_minus_std_and_judge = abs_cliff_mean_plus_minus_std.copy()
for i in range(abs_cliff_mean_plus_minus_std_and_judge.shape[0]):
    for j in range(abs_cliff_mean_plus_minus_std_and_judge.shape[1]):
        mean_plus_minus_std = abs_cliff_mean_plus_minus_std_and_judge.iloc[i, j]
        m = mean_plus_minus_std.split(" ")[0]

        if m != "-":
            judge = judge_cliff(float(m))

            abs_cliff_mean_plus_minus_std_and_judge.iloc[
                i, j
            ] = mean_plus_minus_std + " ({})".format(judge)


print(abs_cliff_mean_plus_minus_std)
print(abs_cliff_mean_plus_minus_std_and_judge)

################################################################################
## Plots abs_cliff_mean_plus_minus_std_and_judge as a colormap
################################################################################
utils.plt.configure_mpl(
    plt,
    dpi=100,
    figsize=(8.7, 8.7),
    fontsize=7,
    legendfontsize="xx-small",
    hide_spines=True,
    tick_size=0.8,
    tick_width=0.2,
)
abs_cliff_mean = pd.DataFrame(data=abs_cliff_mean, index=groups, columns=groups)


fig, ax = plt.subplots()
bounds = [0, 0.147, 0.330, 0.474, 1]
RGBA_colors = [utils.plt.get_RGBA_from_colormap(v) for v in bounds][1:]
cmap = colors.ListedColormap(RGBA_colors)
norm = colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(abs_cliff_mean, cmap=cmap, norm=norm)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(groups)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(groups)
ax.invert_yaxis()
fig.colorbar(im, ax=ax, spacing="proportional", shrink=0.82)
# Makes the frame invisible
for _, spine in ax.spines.items():
    spine.set_visible(False)
# Loop over data dimensions and create text annotations.
for i in range(len(groups)):
    for j in range(len(groups)):
        annot_val = abs_cliff_mean.iloc[i, j]
        annot_color = "black" if annot_val <= bounds[3] else "white"
        annot_text = (
            abs_cliff_mean_plus_minus_std_and_judge.iloc[i, j]
            .replace(" +/- ", "\n+/-\n")
            .replace(" (", "\n(")
        )
        text = ax.text(
            j,
            i,
            annot_text,
            ha="center",
            va="center",
            color=annot_color,
        )
# fig.show()

## Saves
med_mad_df = to_med_mad_df(meds_dict)
spath_csv = utils.general.mk_spath("med_mad/{}.csv".format(args.ftr))
utils.general.save(med_mad_df, spath_csv)

spath_csv = utils.general.mk_spath(
    "abs_cliff_mean_plus_minus_std/{}.csv".format(args.ftr)
)
utils.general.save(abs_cliff_mean_plus_minus_std_and_judge, spath_csv)

spath_tiff = spath_csv.replace(".csv", ".tiff")
utils.general.save(fig, spath_tiff)

# ## Saves
# utils.general.save(
#     abs_cliff_mean_plus_minus_std,
#     "{}_abs_cliff_mean_plus_minus_std.csv".format(args.ftr),
# )
# ## Saves
# utils.general.save(
#     fig,
#     "{}_abs_cliff_mean_plus_minus_std.tiff".format(args.ftr),
# )


# koko


# utils.plt.annotated_heatmap = annotated_heatmap
# from matplotlib import colors

# bounds = [0, 0.147, 0.330, 0.474, 1]
# RGBA_colors = [utils.plt.get_RGBA_from_colormap(v) for v in bounds][1:]
# # cmap = colors.ListedColormap(
# #     ["navy", "royalblue", "lightsteelblue", "beige"],
# # )
# cmap = colors.ListedColormap(RGBA_colors)

# # norm = colors.BoundaryNorm([2, 4, 6, 8], cmap.N - 1)
# norm = colors.BoundaryNorm(bounds, cmap.N)

# fig = utils.plt.annotated_heatmap(abs_cliff_mean, cmap=cmap, norm=norm, labels=groups)
# from matplotlib.cm import ScalarMappable, get_cmap

# # fig.colorbar(get_cmap(fig), ax=fig.axes[0], spacing='proportional')
# # fig.colorbar(ScalarMappable(get_cmap(fig)), ax=fig.axes[0], spacing='uniform')
# fig.colorbar(ScalarMappable(get_cmap(fig)), ax=fig.axes[0], spacing="proportional")
# fig.show()

# import seaborn as sns

# sns.heatmap(abs_cliff_mean, spacing="proportional")


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
