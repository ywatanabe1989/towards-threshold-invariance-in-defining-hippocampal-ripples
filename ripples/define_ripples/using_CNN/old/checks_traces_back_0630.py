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
utils.plt.configure_mpl(plt)


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
lpath_rip_sec = utils.pj.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="CNN_labeled/D{}+".format(args.n_mouse),
)
lpath_rip_magni = utils.pj.path_converters.LFP_to_ripple_magni(lpath_lfp)
lpath_mep_magni = utils.pj.path_converters.LFP_to_MEP_magni(lpath_lfp)

## Gets Parameters
samp_rate = utils.pj.get_samp_rate_int_from_fpath(lpath_lfp)
dt_sec = 1.0 / samp_rate


################################################################################
## Loads
################################################################################
lfp = utils.general.load(lpath_lfp)
rip_sec = utils.general.load(lpath_rip_sec)
rip_magni = utils.general.load(lpath_rip_magni)
mep_magni = utils.general.load(lpath_mep_magni)


## Makes ripple analog signal
rip_analog_sig = np.zeros_like(lfp)
rip_pred_proba_sig = np.zeros_like(lfp)
for i_rip, rip in rip_sec.iterrows():
    start_pts = int(rip["start_sec"] * samp_rate)
    end_pts = int(rip["end_sec"] * samp_rate)
    rip_analog_sig[start_pts:end_pts] = 1
    rip_pred_proba_sig[start_pts:end_pts] = rip["pred_probas_ripple_CNN"]


################################################################################
## Plots
################################################################################
def plot_traces(start_sec=6516, dur_sec=3):
    # start_sec = 6516 # 6514 # 6514 # F2T, T2F
    end_sec = start_sec + dur_sec  # 5 # 3880
    x_sec = np.arange(start_sec, end_sec, dt_sec)  # x

    start_pts = int(start_sec * samp_rate)
    end_pts = int(end_sec * samp_rate)

    lfp_plt = lfp[start_pts:end_pts].squeeze()
    rip_analog_sig_plt = rip_analog_sig[start_pts:end_pts].squeeze()

    # fixme; This magnitude are calculated by another filter.
    #        Thus the time is slided.
    rip_magni_plt = rip_magni[start_pts:end_pts].squeeze()

    # fixme: Therefor, as the above problem,
    #        this is also slided because of the different filter length.
    mep_magni_plt = mep_magni[start_pts:end_pts].squeeze()
    rip_pred_proba_sig_plt = rip_pred_proba_sig[start_pts:end_pts].squeeze()

    ## Gets ripple band LFP
    RIPPLE_CANDI_LIM_HZ = utils.general.load("./conf/global.yaml")[
        "RIPPLE_CANDI_LIM_HZ"
    ]
    # filted_plt, _, _ = utils.dsp.define_ripple_candidates(
    filted_plt, _, _ = utils.pj.define_ripple_candidates(
        x_sec,
        lfp_plt,
        samp_rate,
        lo_hz=RIPPLE_CANDI_LIM_HZ[0],
        hi_hz=RIPPLE_CANDI_LIM_HZ[1],
    )  # 150, 250, # fixme

    # Plot
    linewidth = 1
    dpi = 300
    fig, ax = plt.subplots(5, 1, sharex=True)

    ax[0].plot(
        x_sec,
        rip_pred_proba_sig_plt,
        linewidth=linewidth,
        label="estimated ripple probability",
        color="black",
    )
    ax[1].plot(
        x_sec,
        lfp_plt,
        linewidth=linewidth / 5.0,
        label="raw LFP",
        color="black",
    )
    rip_band_str = "{}-{} Hz".format(RIPPLE_CANDI_LIM_HZ[0], RIPPLE_CANDI_LIM_HZ[1])
    ax[2].plot(
        x_sec,
        filted_plt,
        linewidth=linewidth / 10.0,
        label="ripple band LFP ({})".format(rip_band_str),
        color="black",
    )
    ax[3].plot(
        x_sec,
        rip_magni_plt,
        linewidth=linewidth / 3.0,
        label="ripple band normalized magnitude",
        color="black",
    )
    ax[4].plot(
        x_sec,
        mep_magni_plt,
        linewidth=linewidth / 3.0,
        label="MEP normalized magnitude",
        color="black",
    )

    ## Ripple Coloring
    rip_sec_plt = rip_sec[
        (start_sec < rip_sec["start_sec"]) & (rip_sec["end_sec"] < end_sec)
    ]

    for i in range(len(ax)):
        for ripple in rip_sec_plt.itertuples():
            ax[i].axvspan(
                ripple.start_sec,
                ripple.end_sec - 1.0 / samp_rate,
                alpha=0.1,
                color="red",
                zorder=1000,
            )

    handles_d = utils.general.listed_dict()
    labels_d = utils.general.listed_dict()
    for i in range(len(ax)):
        handles_d[i], labels_d[i] = ax[i].get_legend_handles_labels()
        handles_d[i].append(matplotlib.patches.Patch(facecolor="red", alpha=0.1))
        labels_d[i].append("ripple candi.")
        ax[i].legend(handles_d[i], labels_d[i], loc="upper left")
        ax[i].set_ylabel("Amp. [mV]")

    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Repr. traces")

    ax[1].set_ylim(-1000, 1000)
    ax[2].set_ylim(-200, 200)
    ax[3].set_ylim(0, 10)
    ax[4].set_ylim(0, 10)

    return fig


for _ in range(10):
    try:
        s = np.random.randint(0, 50000)  #
        # s = 25966
        # 39188
        # 41436
        print(s)
        fig = plot_traces(start_sec=s, dur_sec=20)
        # fig.show()
        utils.general.save(
            fig, "/tmp/fake/repr/representative_traces_s_{}.png".format(s)
        )
    except:
        pass
