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
    "-nm", "--n_mouse", default="02", choices=["01", "02", "03", "04", "05"], help=" "
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
utils.plt.configure_mpl(plt)


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


################################################################################
## Parameters
################################################################################
N_ELECTRODES = 5
N_PERIODS_PER_ELECTRODE = 30
DURATION_SEC = args.duration_sec  # 3


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)
## Gets Parameters
SAMP_RATE = utils.pj.get_samp_rate_int_from_fpath(LPATH_HIPPO_LFP_NPY_LIST_MICE[0])


################################################################################
## Main
################################################################################
for _ in range(N_ELECTRODES):
    ## FPATHs
    lpath_lfp = random.choice(LPATH_HIPPO_LFP_NPY_LIST_MICE)
    lpath_rip_magni = utils.pj.path_converters.LFP_to_ripple_magni(lpath_lfp)
    lpath_mep_magni = utils.pj.path_converters.LFP_to_MEP_magni(lpath_lfp)
    # rip_sec
    lpath_rip_sec_CNN = utils.pj.path_converters.LFP_to_ripples(
        lpath_lfp,
        rip_sec_ver="CNN_labeled/D{}-".format(args.n_mouse),
    )
    lpath_rip_sec_CNN = utils.pj.path_converters.cycle_dataset(
        lpath_rip_sec_CNN, args.n_mouse
    )
    lpath_rip_sec_GMM = lpath_rip_sec_CNN.replace("CNN", "GMM")

    ################################################################################
    ## Loads
    ################################################################################
    lfp = utils.general.load(lpath_lfp)
    rip_magni = utils.general.load(lpath_rip_magni)
    mep_magni = utils.general.load(lpath_mep_magni)
    # rip_sec
    _rip_sec_CNN = utils.general.load(lpath_rip_sec_CNN)
    _rip_sec_GMM = utils.general.load(lpath_rip_sec_GMM)
    rip_sec = _rip_sec_CNN[["start_sec", "end_sec", "pred_probas_ripple_CNN"]]
    # makes X2X column
    rip_sec["X2X"] = np.nan  # init
    are_T2T = _rip_sec_GMM["are_ripple_GMM"] & _rip_sec_CNN["are_ripple_CNN"].astype(
        bool
    )
    are_F2T = ~_rip_sec_GMM["are_ripple_GMM"] & _rip_sec_CNN["are_ripple_CNN"].astype(
        bool
    )
    are_T2F = _rip_sec_GMM["are_ripple_GMM"] & ~_rip_sec_CNN["are_ripple_CNN"].astype(
        bool
    )
    are_F2F = ~_rip_sec_GMM["are_ripple_GMM"] & ~_rip_sec_CNN["are_ripple_CNN"].astype(
        bool
    )
    rip_sec["X2X"][are_T2T] = "T2T"
    rip_sec["X2X"][are_F2T] = "F2T"
    rip_sec["X2X"][are_T2F] = "T2F"
    rip_sec["X2X"][are_F2F] = "F2F"

    ################################################################################
    ## Makes ripple analog signal
    ################################################################################
    rip_analog_sig = np.zeros_like(lfp)
    rip_pred_proba_sig = np.zeros_like(lfp)
    for i_rip, rip in rip_sec.iterrows():
        start_pts = int(rip["start_sec"] * SAMP_RATE)
        end_pts = int(rip["end_sec"] * SAMP_RATE)
        rip_analog_sig[start_pts:end_pts] = 1
        rip_pred_proba_sig[start_pts:end_pts] = rip["pred_probas_ripple_CNN"]

    ################################################################################
    ## Plots
    ################################################################################
    signals_dict = {
        "lfp": lfp.squeeze(),
        "rip_analog_sig": rip_analog_sig.squeeze(),
        "rip_magni": rip_magni.squeeze(),
        "mep_magni": mep_magni.squeeze(),
        "rip_sec": rip_sec,
        "rip_pred_proba_sig": rip_pred_proba_sig.squeeze(),
    }

    for _ in range(N_PERIODS_PER_ELECTRODE):
        rand_start_sec = np.random.randint(len(lfp) / SAMP_RATE - DURATION_SEC - 1)
        # fig = utils.pj.plt.plot_signals(signals_dict, lpath_lfp, start_sec=s, dur_sec=d)
        fig, out_sig = utils.pj.plot_traces_X2X(
            signals_dict, lpath_lfp, start_sec=rand_start_sec, dur_sec=DURATION_SEC
        )
        sfname_png = lpath_lfp.replace("./", "|").replace("/", "|")
        sfname_png = sfname_png + "_start_{}_sec.png".format(rand_start_sec)
        spath_png = utils.general.mk_spath(
            "mouse#{}/duration_{}_sec/fig/{}".format(
                args.n_mouse, DURATION_SEC, sfname_png
            )
        )
        utils.general.save(fig, spath_png)

        spath_csv = spath_png.replace(".png", ".csv").replace("/fig/", "/csv/")
        utils.general.save(out_sig, spath_csv)
        # spath = "./ripples/define_ripples/using_CNN/checks_traces/mouse#{}/duration_{}_sec/{}".format(
        #     args.n_mouse, DURATION_SEC, sfname
        # )

        plt.close()

# python3 ./ripples/define_ripples/using_CNN/checks_traces.py

## EOF
