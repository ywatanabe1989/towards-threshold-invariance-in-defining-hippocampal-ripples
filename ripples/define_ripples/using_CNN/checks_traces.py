#!/usr/bin/env python3

import argparse
import os
import random
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(".")
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
ap.add_argument("-dur", "--duration_sec", default=3, type=int, help=" ")
args = ap.parse_args()


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np, random=random)


################################################################################
## Configures matplotlib
################################################################################
utils.plt.configure_mpl(plt, figsize=(18.1, 12.0), fontsize=8, legendfontsize="small")


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## Parameters
################################################################################
N_ELECTRODES = -1  # 2  # 5
N_PERIODS_PER_ELECTRODE = 3  # 30
DURATION_SEC = args.duration_sec


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)
## Gets Parameters
SAMP_RATE = utils.pj.get_samp_rate_int_from_fpath(LPATH_HIPPO_LFP_NPY_LIST_MICE[0])
LPATH_LFP_ELECTRODES = np.random.permutation(LPATH_HIPPO_LFP_NPY_LIST_MICE)[
    :N_ELECTRODES
]

################################################################################
## Main
################################################################################
for lpath_lfp in LPATH_LFP_ELECTRODES:
    # lpath_lfp = LPATH_LFP_ELECTRODES[0]
    ## FPATHs
    # lpath_lfp = random.choice(LPATH_HIPPO_LFP_NPY_LIST_MICE)
    lpath_rip_magni = utils.pj.path_converters.LFP_to_ripple_magni(lpath_lfp)
    lpath_mep_magni = utils.pj.path_converters.LFP_to_MEP_magni(lpath_lfp)

    ################################################################################
    ## Loads
    ################################################################################
    lfp = utils.general.load(lpath_lfp)
    rip_magni = utils.general.load(lpath_rip_magni)
    mep_magni = utils.general.load(lpath_mep_magni)
    # rip_sec
    rip_sec = utils.pj.load.rip_sec(
        lpath_lfp,
        rip_sec_ver="CNN_labeled/D{}-".format(args.n_mouse),
        cycle_dataset=True,
        n_mouse=args.n_mouse,
    )

    # makes X2X column
    rip_sec["X2X"] = np.nan  # init
    are_T2T = rip_sec["are_ripple_GMM"] & rip_sec["are_ripple_CNN"].astype(bool)
    are_F2T = ~rip_sec["are_ripple_GMM"] & rip_sec["are_ripple_CNN"].astype(bool)
    are_T2F = rip_sec["are_ripple_GMM"] & ~rip_sec["are_ripple_CNN"].astype(bool)
    are_F2F = ~rip_sec["are_ripple_GMM"] & ~rip_sec["are_ripple_CNN"].astype(bool)

    rip_sec.loc[are_T2T, "X2X"] = "T2T"
    rip_sec.loc[are_F2T, "X2X"] = "F2T"
    rip_sec.loc[are_T2F, "X2X"] = "T2F"
    rip_sec.loc[are_F2F, "X2X"] = "F2F"

    """
    Regarding 'psx_ripple' for X2X groups, the result below seems wrong.
    # print(rip_sec["psx_ripple"][rip_sec["X2X"] == "T2T"].min()) # 0.29
    # print(rip_sec["psx_ripple"][rip_sec["X2X"] == "F2F"].max()) # 0.66
    However, it's OK. 'psx_ripple' is calculated for each fold and
    the threshold for them are independently determined
    by the cleanlab, a package for Confident Learning.
    """

    ################################################################################
    ## Makes ripple analog signal
    ################################################################################
    rip_analog_sig = np.zeros_like(lfp)
    rip_analog_sig_T2T = np.zeros_like(lfp)
    rip_analog_sig_F2T = np.zeros_like(lfp)
    rip_analog_sig_T2F = np.zeros_like(lfp)
    rip_analog_sig_F2F = np.zeros_like(lfp)
    rip_pred_proba_sig = np.zeros_like(lfp)
    for i_rip, rip in rip_sec.iterrows():
        start_pts = int(rip["start_sec"] * SAMP_RATE)
        end_pts = int(rip["end_sec"] * SAMP_RATE)
        rip_analog_sig[start_pts:end_pts] = 1
        rip_analog_sig_T2T[start_pts:end_pts] = 1 if rip["X2X"] == "T2T" else 0
        rip_analog_sig_F2T[start_pts:end_pts] = 1 if rip["X2X"] == "F2T" else 0
        rip_analog_sig_T2F[start_pts:end_pts] = 1 if rip["X2X"] == "T2F" else 0
        rip_analog_sig_F2F[start_pts:end_pts] = 1 if rip["X2X"] == "F2F" else 0
        rip_pred_proba_sig[start_pts:end_pts] = rip["psx_ripple"]

    ################################################################################
    ## Plots
    ################################################################################
    input_signals_dict = {
        "lfp": lfp.squeeze(),
        "rip_analog_sig": rip_analog_sig.squeeze(),
        "rip_analog_sig_T2T": rip_analog_sig_T2T.squeeze(),
        "rip_analog_sig_F2T": rip_analog_sig_F2T.squeeze(),
        "rip_analog_sig_T2F": rip_analog_sig_T2F.squeeze(),
        "rip_analog_sig_F2F": rip_analog_sig_F2F.squeeze(),
        "rip_magni": rip_magni.squeeze(),
        "mep_magni": mep_magni.squeeze(),
        "rip_sec": rip_sec,
        "rip_pred_proba_sig": rip_pred_proba_sig.squeeze(),
    }

    for _ in range(N_PERIODS_PER_ELECTRODE):
        rand_start_sec = np.random.randint(len(lfp) / SAMP_RATE - DURATION_SEC - 1)
        fig, out_sig = utils.pj.plot_traces_X2X(
            input_signals_dict,
            lpath_lfp,
            start_sec=rand_start_sec,
            dur_sec=DURATION_SEC,
        )

        # fig.show()

        sfname_tiff = lpath_lfp.replace("./", "|").replace("/", "|")
        sfname_tiff = sfname_tiff + "_start_{}_sec.tiff".format(rand_start_sec)
        spath_tiff = utils.general.mk_spath(
            "mouse#{}/duration_{}_sec/fig/{}".format(
                args.n_mouse, DURATION_SEC, sfname_tiff
            )
        )
        utils.general.save(fig, spath_tiff)

        spath_csv = spath_tiff.replace(".tiff", ".csv").replace("/fig/", "/csv/")
        utils.general.save(out_sig, spath_csv)

        plt.close()

# python3 ./ripples/define_ripples/using_CNN/checks_traces.py

## EOF
