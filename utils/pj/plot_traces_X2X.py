#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils


def plot_traces_X2X(signals_dict, lpath_lfp, start_sec=6516, dur_sec=3, samp_rate=1000):
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
    lfp_plt = signals_dict["lfp"][start_pts:end_pts]
    filted_plt, _, _ = utils.pj.define_ripple_candidates(
        x_sec,
        lfp_plt,
        samp_rate,
        lo_hz=RIP_LO_HZ,
        hi_hz=RIP_HI_HZ,
    )
    rip_analog_sig_plt = signals_dict["rip_analog_sig"][start_pts:end_pts]
    rip_analog_sig_T2T_plt = signals_dict["rip_analog_sig_T2T"][start_pts:end_pts]
    rip_analog_sig_F2T_plt = signals_dict["rip_analog_sig_F2T"][start_pts:end_pts]
    rip_analog_sig_T2F_plt = signals_dict["rip_analog_sig_T2F"][start_pts:end_pts]
    rip_analog_sig_F2F_plt = signals_dict["rip_analog_sig_F2F"][start_pts:end_pts]
    rip_magni_plt = signals_dict["rip_magni"][start_pts:end_pts]
    mep_magni_plt = signals_dict["mep_magni"][start_pts:end_pts]
    rip_pred_proba_sig_plt = signals_dict["rip_pred_proba_sig"][start_pts:end_pts]

    ## Puts all data into one dataframe
    index = [
        "ripple confidence",
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
    df.loc["ripple confidence", "signal"] = rip_pred_proba_sig_plt
    df.loc["raw LFP", "signal"] = lfp_plt
    df.loc[
        "ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "signal"
    ] = filted_plt
    df.loc["ripple band normalized magnitude", "signal"] = rip_magni_plt
    df.loc["MEP normalized magnitude", "signal"] = mep_magni_plt

    # ylim
    df.loc["ripple confidence", "ylim"] = (0, 1.05)
    df.loc["raw LFP", "ylim"] = (-1000, 1000)
    df.loc["ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "ylim"] = (
        -200,
        200,
    )
    df.loc["ripple band normalized magnitude", "ylim"] = (0, 15)
    df.loc["MEP normalized magnitude", "ylim"] = (0, 15)

    # linewidth
    df.loc["ripple confidence", "linewidth"] = 1
    df.loc["raw LFP", "linewidth"] = 1.0 / 5
    df.loc["ripple band LFP ({}-{} Hz)".format(RIP_LO_HZ, RIP_HI_HZ), "linewidth"] = (
        1.0 / 5
    )
    df.loc["ripple band normalized magnitude", "linewidth"] = 1.0 / 5
    df.loc["MEP normalized magnitude", "linewidth"] = 1.0 / 5

    ## Plots
    fig, axes = plt.subplots(df.shape[0], 1, sharex=True)
    out_sig = pd.DataFrame({"x_sec": x_sec})
    legends_1 = []
    for i_ax, ax in enumerate(axes):
        row = df.iloc[i_ax, :]
        label = row.name
        ax.plot(x_sec, row.signal, linewidth=row.linewidth, label=label, color="black")
        ax.set_ylim(row.ylim)
        leg1 = ax.legend(
            loc="upper left",
            framealpha=0,
            edgecolor=None,
            facecolor=None,
            shadow=False,
        )
        legends_1.append(leg1)  # for multiple legends; here, trace only
        out_sig[label] = row.signal  # to save

    title = "file: {}".format(lpath_lfp)
    fig.suptitle(title)
    axes[-1].set_xlabel("Time [sec]")
    fig.text(0.02, 0.5, "Amplitude [$\mu$V]", va="center", rotation="vertical")

    ################################################################################
    ## Fills ripple periods
    ################################################################################
    rip_sec = signals_dict["rip_sec"]
    rip_sec_plt = signals_dict["rip_sec"][
        (start_sec < signals_dict["rip_sec"]["start_sec"])
        & (rip_sec["end_sec"] < end_sec)
    ]

    for i_ax, ax in enumerate(axes):
        for ripple in rip_sec_plt.itertuples():
            c = utils.plt.colors.to_RGBA(
                utils.general.load("./conf/global.yaml")["GROUP_COLORS"][ripple.X2X],
                alpha=0.3,
            )

            ax.axvspan(
                ripple.start_sec,
                ripple.end_sec - 1.0 / samp_rate,
                color=c,
                zorder=1000,
            )

        if i_ax == 0:
            ## multiple legends; out of the figure
            handles, labels = [], []
            for x2x in ["T2T", "F2T", "T2F", "F2F"]:
                c = utils.plt.colors.to_RGBA(
                    utils.general.load("./conf/global.yaml")["GROUP_COLORS"][x2x],
                    alpha=0.3,
                )
                handles.append(matplotlib.patches.Patch(facecolor=c, alpha=0.1))
                labels.append(x2x)

            leg2 = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
            ax.add_artist(legends_1[i_ax])

        # else:
        #     import pdb

        #     pdb.set_trace()
        #     ax.legend()

    out_sig["rip_analog_sig_T2T_plt"] = rip_analog_sig_T2T_plt
    out_sig["rip_analog_sig_F2T_plt"] = rip_analog_sig_F2T_plt
    out_sig["rip_analog_sig_T2F_plt"] = rip_analog_sig_T2F_plt
    out_sig["rip_analog_sig_F2F_plt"] = rip_analog_sig_F2F_plt

    return fig, out_sig