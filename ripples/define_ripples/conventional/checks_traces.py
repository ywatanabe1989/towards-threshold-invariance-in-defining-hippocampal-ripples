#!/usr/bin/env python3

import argparse
import math
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(".")
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-n",
    "--npy_fpath",
    default="./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy",
    help="The path of the input lfp file (.npy)",
)
args = ap.parse_args()


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


## PATHs
lpath_lfp = args.npy_fpath

## Loads
lfp = utils.general.load(lpath_lfp).squeeze()


## Parameters
samp_rate = utils.pj.get_samp_rate_int_from_fpath(lpath_lfp)
step_sec = 1.0 / samp_rate


## Limit LFP regarding time
start_sec, end_sec = 0, 2000
time_x_sec = np.arange(start_sec, end_sec, step_sec)

start_pts, end_pts = int(start_sec * samp_rate), int(end_sec * samp_rate)
lfp = lfp[start_pts:end_pts]


## Detect Ripple Candidates
print(
    "\nDetecting ripples from {} (Length: {:.1f}h)\n".format(
        lpath_lfp, len(lfp) / samp_rate / 3600
    )
)
RIPPLE_CANDI_LIM_HZ = utils.general.load("./conf/global.yaml")["RIPPLE_CANDI_LIM_HZ"]

zscore_thres = 1
filted, ripple_magni, rip_sec = utils.pj.define_ripple_candidates(
    time_x_sec,
    lfp[:, np.newaxis].astype(np.float32),
    samp_rate,
    lo_hz=RIPPLE_CANDI_LIM_HZ[0],
    hi_hz=RIPPLE_CANDI_LIM_HZ[1],
    zscore_threshold=zscore_thres,
)  # 150, 250

################################################################################
## Plots
################################################################################
utils.plt.configure_mpl(plt)

# slicing
start_sec_plt = 1175.8
end_sec_plt = start_sec_plt + 2.2  # 1178.0
x_plt = np.arange(start_sec_plt + step_sec, end_sec_plt, step_sec)

ss = math.floor(start_sec_plt * samp_rate)
ee = math.floor(end_sec_plt * samp_rate)

lfp_plt = lfp[ss:ee]
filted_plt = filted.squeeze()[ss:ee]
ripple_magni_plt = ripple_magni.squeeze()[ss:ee]
ripple_magni_std_plt = np.ones_like(ripple_magni_plt) * ripple_magni.std()

# Plots lines
fig, ax = plt.subplots(3, 1, sharex=True)
lw = 1
ax[0].plot(x_plt, lfp_plt, linewidth=lw, label="raw LFP")
label_filted = "ripple band LFP ({}-{} Hz)".format(
    RIPPLE_CANDI_LIM_HZ[0], RIPPLE_CANDI_LIM_HZ[1]
)
ax[1].plot(x_plt, filted_plt, linewidth=lw, label=label_filted)
ax[2].plot(x_plt, ripple_magni_plt, linewidth=lw, label="ripple band magnitude")
ax[2].plot(x_plt, ripple_magni_std_plt, linewidth=lw, label="1 SD")

# Areas indicating ripple candidates
rip_sec_plt = rip_sec[
    (start_sec_plt < rip_sec["start_sec"]) & (rip_sec["end_sec"] < end_sec_plt)
]

for i in range(len(ax)):
    for ripple in rip_sec_plt.itertuples():
        ax[i].axvspan(
            ripple.start_sec,
            ripple.end_sec,
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


ax[0].legend(handles_d[0], labels_d[0], loc="upper left")
ax[0].set_ylabel("Amplitude [uV]")
ax[0].set_title("Defining Ripple Candidates")
ax[1].legend(handles_d[1], labels_d[1], loc="upper left")
ax[1].set_ylabel("Amplitude [uV]")
ax[2].legend(handles_d[2], labels_d[2], loc="upper left")
ax[2].set_ylabel("Amplitude [uV]")
ax[2].set_xlabel("Time [sec]")
# plt.show()
spath_tiff = "traces_{}-{}_sec.tiff".format(start_sec_plt, end_sec_plt)
utils.general.save(plt, spath_tiff)


rip_sec_plt_digi = np.zeros_like(x_plt)
x_plt_start = int(x_plt[0] * samp_rate)
for ripple in rip_sec_plt.itertuples():
    rip_sec_plt_digi[
        int(ripple.start_sec * samp_rate)
        - x_plt_start : int(ripple.end_sec * samp_rate)
        - x_plt_start
    ] = 1


################################################################################
## Saves
################################################################################
df = pd.DataFrame(
    {
        "x_plt": x_plt.squeeze(),
        "lfp_plt": lfp_plt.squeeze(),
        "filted_plt": filted_plt.squeeze(),
        "ripple_magni_plt": ripple_magni_plt.squeeze(),
        "ripple_magni_std_plt": ripple_magni_std_plt.squeeze(),
        "ripple_digi": rip_sec_plt_digi,
    }
)
spath_csv = spath_tiff.replace(".tiff", ".csv")
utils.general.save(df, spath_csv)

# python3 ./ripples/define_ripples/conventional/checks_traces.py


## EOF
