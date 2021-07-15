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


def plot_signals(signals_dict, start_sec=6516, dur_sec=3, samp_rate=1000):
    ## Gets ripple band LFP
    RIPPLE_CANDI_LIM_HZ = utils.general.load("./conf/global.yaml")[
        "RIPPLE_CANDI_LIM_HZ"
    ]
    RIP_LO_HZ = RIPPLE_CANDI_LIM_HZ[0]
    RIP_HI_HZ = RIPPLE_CANDI_LIM_HZ[1]

    end_sec = start_sec + dur_sec
    x_sec = np.arange(start_sec, end_sec, 1.0 / samp_rate)

    start_pts = int(start_sec * samp_rate)
    end_pts = int(end_sec * samp_rate)

    ## Signals
    # lfp_plt = lfp[start_pts:end_pts].squeeze()
    lfp_plt = signals_dict["lfp"][start_pts:end_pts]
    filted_plt, _, _ = utils.pj.define_ripple_candidates(
        x_sec,
        lfp_plt,
        samp_rate,
        lo_hz=RIP_LO_HZ,
        hi_hz=RIP_HI_HZ,
    )
    # rip_analog_sig_plt = rip_analog_sig[start_pts:end_pts].squeeze()
    # rip_magni_plt = rip_magni[start_pts:end_pts].squeeze()
    # mep_magni_plt = mep_magni[start_pts:end_pts].squeeze()
    # rip_pred_proba_sig_plt = rip_pred_proba_sig[start_pts:end_pts].squeeze()
    rip_analog_sig_plt = signals_dict["rip_analog_sig"][start_pts:end_pts]
    rip_magni_plt = signals_dict["rip_magni"][start_pts:end_pts]
    mep_magni_plt = signals_dict["mep_magni"][start_pts:end_pts]
    rip_pred_proba_sig_plt = signals_dict["rip_pred_proba_sig"][start_pts:end_pts]

    ## Puts all data into one dataframe
    index = [
        "estimated ripple probability",
        "raw LFP",
        "ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ),
        "ripple band normalized magnitude",
        "MEP normalized magnitude",
    ]

    columns = [
        "signal",
        "ylim",
        "linewidth",
    ]

    df = pd.DataFrame(index=index, columns=columns)

    # signal
    df.loc["estimated ripple probability", "signal"] = rip_pred_proba_sig_plt
    df.loc["raw LFP", "signal"] = lfp_plt
    df.loc[
        "ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "signal"
    ] = filted_plt
    df.loc["ripple band normalized magnitude", "signal"] = rip_magni_plt
    df.loc["MEP normalized magnitude", "signal"] = mep_magni_plt

    # ylim
    df.loc["estimated ripple probability", "ylim"] = (0, 1.05)
    df.loc["raw LFP", "ylim"] = (-1000, 1000)
    df.loc["ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "ylim"] = (
        -200,
        200,
    )
    df.loc["ripple band normalized magnitude", "ylim"] = (0, 10)
    df.loc["MEP normalized magnitude", "ylim"] = (0, 10)
    # df.loc["ripple band normalized magnitude", "ylim"] = (0.1, 10) # fixme; log
    # df.loc["MEP normalized magnitude", "ylim"] = (0.1, 10)

    # # yscale
    # df.loc["estimated ripple probability", "yscale"] = None
    # df.loc["raw LFP", "yscale"] = None
    # df.loc["ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "yscale"] = None
    # df.loc["ripple band normalized magnitude", "yscale"] = "log"
    # df.loc["MEP normalized magnitude", "yscale"] = "log"

    # linewidth
    df.loc["estimated ripple probability", "linewidth"] = 1
    df.loc["raw LFP", "linewidth"] = 1.0 / 5
    df.loc["ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "linewidth"] = (
        1.0 / 5
    )
    df.loc["ripple band normalized magnitude", "linewidth"] = 1.0 / 5
    df.loc["MEP normalized magnitude", "linewidth"] = 1.0 / 5

    ## Plots
    fig, axes = plt.subplots(df.shape[0], 1, sharex=True)
    for i_ax, ax in enumerate(axes):
        row = df.iloc[i_ax, :]
        ax.set_ylim(row.ylim)
        ax.plot(
            x_sec, row.signal, linewidth=row.linewidth, label=row.name, color="black"
        )
        ax.legend(loc="upper left")
        ax.set_ylabel("Amp. [uV]")

        # try:
        #     ax.set_yscale(row.yscale)
        # except:
        #     pass

    ################################################################################
    ## Fills ripple periods
    ################################################################################
    rip_sec_plt = rip_sec[
        (start_sec < rip_sec["start_sec"]) & (rip_sec["end_sec"] < end_sec)
    ]

    for ax in axes:
        for ripple in rip_sec_plt.itertuples():

            c = utils.plt.colors.to_RGBA(
                utils.general.load("./conf/global.yaml")["GROUP_COLORS"][ripple.X2X],
                alpha=0.3,
            )

            ax.axvspan(
                ripple.start_sec,
                ripple.end_sec - 1.0 / samp_rate,
                color=c,
                # alpha=0.1,
                # color="red",
                zorder=1000,
            )

        handles, labels = ax.get_legend_handles_labels()
        for x2x in ["T2T", "F2T", "T2F", "F2F"]:
            c = utils.plt.colors.to_RGBA(
                utils.general.load("./conf/global.yaml")["GROUP_COLORS"][x2x],
                alpha=0.3,
            )
            # handles.append(matplotlib.patches.Patch(facecolor=c, alpha=0.1))
            handles.append(matplotlib.patches.Patch(facecolor=c, alpha=0.1))
            labels.append(x2x)
            ax.legend(handles, labels, loc="upper left")

    # fig.show()
    return fig


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
lpath_rip_sec_CNN = utils.pj.path_converters.LFP_to_ripples(
    lpath_lfp,
    rip_sec_ver="CNN_labeled/D{}+".format(args.n_mouse),
)
lpath_rip_sec_GMM = lpath_rip_sec_CNN.replace("CNN", "GMM")
lpath_rip_magni = utils.pj.path_converters.LFP_to_ripple_magni(lpath_lfp)
lpath_mep_magni = utils.pj.path_converters.LFP_to_MEP_magni(lpath_lfp)

## Gets Parameters
samp_rate = utils.pj.get_samp_rate_int_from_fpath(lpath_lfp)


################################################################################
## Loads
################################################################################
lfp = utils.general.load(lpath_lfp)
_rip_sec_CNN = utils.general.load(lpath_rip_sec_CNN)
_rip_sec_GMM = utils.general.load(lpath_rip_sec_GMM)
rip_sec = _rip_sec_CNN[["start_sec", "end_sec", "pred_probas_ripple_CNN"]]
rip_sec["X2X"] = np.nan
are_T2T = _rip_sec_GMM["are_ripple_GMM"] & _rip_sec_CNN["are_ripple_CNN"]
are_F2T = ~_rip_sec_GMM["are_ripple_GMM"] & _rip_sec_CNN["are_ripple_CNN"]
are_T2F = _rip_sec_GMM["are_ripple_GMM"] & ~_rip_sec_CNN["are_ripple_CNN"]
are_F2F = ~_rip_sec_GMM["are_ripple_GMM"] & ~_rip_sec_CNN["are_ripple_CNN"]
rip_sec["X2X"][are_T2T] = "T2T"
rip_sec["X2X"][are_F2T] = "T2F"
rip_sec["X2X"][are_T2F] = "T2F"
rip_sec["X2X"][are_F2F] = "F2F"
rip_magni = utils.general.load(lpath_rip_magni)
mep_magni = utils.general.load(lpath_mep_magni)


################################################################################
## Makes ripple analog signal
################################################################################
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
start_sec = 6516
dur_sec = 3
samp_rate = 1000

signals_dict = {
    "lfp": lfp.squeeze(),
    "rip_analog_sig": rip_analog_sig.squeeze(),
    "rip_magni": rip_magni.squeeze(),
    "mep_magni": mep_magni.squeeze(),
    "rip_pred_proba_sig": rip_pred_proba_sig.squeeze(),
}


s, d = 56, 10
s = np.random.randint(5000)
fig = plot_signals(signals_dict, start_sec=s, dur_sec=d)
fig.show()
