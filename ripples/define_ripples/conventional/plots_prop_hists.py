#!/usr/bin/env python
import argparse
import sys

sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--num_mouse", default="01", help=" ")
args = ap.parse_args()


## Configure Matplotlib
utils.plt.configure_mpl(plt)


## PATHs
hipp_lfp_paths_npy = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
hipp_lfp_paths_npy_mouse_i = utils.general.grep(hipp_lfp_paths_npy, args.num_mouse)[1]


## Loads
rip_sec_df = pd.concat(
    [
        utils.general.load(
            utils.pj.path_converters.LFP_to_ripples(f, rip_sec_ver="candi_with_props")
        )
        for f in hipp_lfp_paths_npy_mouse_i
    ]
)


## Excludes columns
rip_sec_df = rip_sec_df[
    ["ln(duration_ms)", "ln(mean MEP magni. / SD)", "ln(ripple peak magni. / SD)"]
]

## Plots
hist_df = pd.DataFrame()
plot = True
if plot:
    ################################################################################
    ## Sets a figure
    ################################################################################
    n_bins = 600
    fig, ax = plt.subplots(1, 3)
    plt.title("Mouse#{}".format(args.num_mouse))

    ################################################################################
    ## 'ln(duration_ms)'
    ################################################################################
    key = "ln(duration_ms)"
    counts, bin_edges, patches = ax[0].hist(
        rip_sec_df[key],
        range=(2.7, 9.3),
        bins=n_bins,
    )

    bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
    hist_df[key + "_bin_centers"] = bin_centers
    hist_df[key + "_counts"] = counts

    ax[0].set_ylim(0, 65000)
    ax[0].set_xlabel("ln(Duration) [a.u.]")

    ################################################################################
    ## 'ln(mean MEP magni. / SD)'
    ################################################################################
    key = "ln(mean MEP magni. / SD)"
    counts, bin_edges, patches = ax[1].hist(
        rip_sec_df[key],
        range=(-3.3, 3.3),
        bins=n_bins,
    )

    bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
    hist_df[key + "_bin_centers"] = bin_centers
    hist_df[key + "_counts"] = counts

    ax[1].set_ylim(0, 40000)
    ax[1].set_xlabel("ln(Mean normalized magnitude of MEP) [a.u.]")

    ################################################################################
    ## 'ln(ripple peak magni. / SD)'
    ################################################################################
    key = "ln(ripple peak magni. / SD)"
    counts, bin_edges, patches = ax[2].hist(
        rip_sec_df[key],
        range=(-0.7, 2.7),
        bins=n_bins,
    )

    bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
    hist_df[key + "_bin_centers"] = bin_centers
    hist_df[key + "_counts"] = counts

    ax[2].set_ylim(0, 40000)
    ax[2].set_xlabel("ln(Normalized ripple peak magnitude) [a.u.]")

    ################################################################################
    ## Plots
    ################################################################################
    # plt.show()
    spath = "props_hists.png"
    utils.general.save(plt, spath)


## Saves
utils.general.save(hist_df, "hist_df_mouse_{n}.csv".format(n=args.num_mouse))

# utils.general.load('/tmp/fake/rip_sec_df_mouse_01_10_perc.csv')

## EOF
