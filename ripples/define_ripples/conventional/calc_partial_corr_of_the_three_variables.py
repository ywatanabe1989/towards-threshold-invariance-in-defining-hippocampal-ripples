#!/usr/bin/env python

import sys

sys.path.append(".")
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils


## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Configure Matplotlib
utils.plt.configure_mpl(plt)


## Parameters
FTR_X = "ln(duration_ms)"
FTR_Y = "ln(ripple peak magni. / SD)"
FTR_Z = "ln(mean MEP magni. / SD)"


corrs = utils.general.listed_dict(["XY_Z", "YZ_X", "ZX_Y"])
ns = []
for n_mouse in ["01", "02", "03", "04", "05"]:
    ## PATHs
    PATHs_HIPP_LFP_NPY_MOUSE = utils.pj.load.get_hipp_lfp_fpaths([n_mouse])

    ## Loads
    rips_df_list = utils.pj.load.rips_sec(
        PATHs_HIPP_LFP_NPY_MOUSE, rip_sec_ver="candi_with_props"
    )

    # ## Samples the same number of data from each mouse
    # indi_mice = [
    #     int(p.split("./data/okada/")[1].split("/day")[0]) - 1 for p in PATHs_HIPP_LFP_NPY_MOUSE
    # ]
    # indi = np.hstack(
    #     [
    #         [i_mouse for _ in range(len(rips_df_list[i_lfp]))]
    #         for i_lfp, i_mouse in enumerate(indi_mice)
    #     ]
    # )

    rips_sec_df = pd.concat(rips_df_list)
    ns.append(len(rips_sec_df))

    rips_sec_df = rips_sec_df[[FTR_X, FTR_Y, FTR_Z]]
    # rips_sec_df["i_mouse"] = indi

    # indi_under_sampled = utils.ml.under_sample(indi)
    # rips_sec_df_under_sampled = rips_sec_df.iloc[indi_under_sampled]
    # print(np.unique(rips_sec_df_under_sampled["i_mouse"], return_counts=True))

    ## Main
    corr_XY_Z = utils.stats.calc_partial_corr(
        np.exp(rips_sec_df[FTR_X] + 1e-5),
        np.exp(rips_sec_df[FTR_Y] + 1e-5),
        np.exp(rips_sec_df[FTR_Z] + 1e-5),
    )  # +1e-5
    print("Corr_XY_Z: {:.3f}".format(corr_XY_Z))  # 0.230
    corrs["XY_Z"].append(corr_XY_Z)

    corr_YZ_X = utils.stats.calc_partial_corr(
        np.exp(rips_sec_df[FTR_Y]),
        np.exp(rips_sec_df[FTR_Z]),
        np.exp(rips_sec_df[FTR_X]),
    )
    print("Corr_YZ_X: {:.3f}".format(corr_YZ_X))  # 0.148
    corrs["YZ_X"].append(corr_YZ_X)

    corr_ZX_Y = utils.stats.calc_partial_corr(
        np.exp(rips_sec_df[FTR_Z]),
        np.exp(rips_sec_df[FTR_X]),
        np.exp(rips_sec_df[FTR_Y]),
    )
    print("Corr_ZX_Y: {:.3f}".format(corr_ZX_Y))  # 0.527
    corrs["ZX_Y"].append(corr_ZX_Y)


def calc_mean_and_std(listed_scalars):
    m = np.mean(listed_scalars)
    s = np.std(listed_scalars)
    return m, s


def pack_to_df(listed_scalars):
    m, s = calc_mean_and_std(listed_scalars)
    index = ["Mean", "Std.", *np.arange(len(listed_scalars))]
    return pd.DataFrame(data=[m, s, *listed_scalars], index=index)


print("X: {}".format(FTR_X))
print("Y: {}".format(FTR_Y))
print("Z: {}".format(FTR_Z))
print(pack_to_df(corrs["XY_Z"]))
print(pack_to_df(corrs["YZ_X"]))
print(pack_to_df(corrs["ZX_Y"]))


## EOF
n = len(rips_sec_df)
print(ns)
