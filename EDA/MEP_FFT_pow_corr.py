#!/usr/bin/env python


import argparse
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
# from scipy import fftpack, stats
from scipy import stats

sys.path.append(".")

import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
args = ap.parse_args()

## Sets tee
# sys.stdout, sys.stderr = utils.general.tee(sys, SDIR + "out")
sys.stdout, sys.stderr = utils.general.tee(sys)

## Parameters
SDIR = "./EDA/MEP_FFT_pow_corr/"


## Configure Matplotlib
utils.plt.configure_mpl(plt)


## Fixes random seed
utils.general.fix_seeds(seed=42, np=np)


## Funcs
# def calc_fft_powers(x, index):
#     X = fftpack.fft(x)
#     powers = np.abs(X)[tuple(index)]
#     return powers


def take_corr(x=None, y=None):
    return np.corrcoef(x, y)[0, 1]


## Parameters
SAMP_RATE = 1000
WINDOW_SIZE_PTS = 1024


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = utils.general.grep(
    LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse
)[1]
LPATH_TRAPE_MEP_NPY_LIST_MOUSE = [
    utils.pj.path_converters.LFP_to_MEP_magni(f) for f in LPATH_HIPPO_LFP_NPY_LIST_MOUSE
]


## Loads
lfps = [utils.general.load(f).squeeze() for f in LPATH_HIPPO_LFP_NPY_LIST_MOUSE]
meps = [utils.general.load(f).squeeze() for f in LPATH_TRAPE_MEP_NPY_LIST_MOUSE]


## To segments
random_starts_pts = np.random.randint(0, high=WINDOW_SIZE_PTS, size=len(lfps))
lfps = np.vstack(
    [
        skimage.util.view_as_windows(
            lfp[random_start:], WINDOW_SIZE_PTS, WINDOW_SIZE_PTS
        )
        for lfp, random_start in zip(lfps, random_starts_pts)
    ]
)
meps = np.vstack(
    [
        skimage.util.view_as_windows(
            mep[random_start:], WINDOW_SIZE_PTS, WINDOW_SIZE_PTS
        )
        for mep, random_start in zip(meps, random_starts_pts)
    ]
)


## Calculates FFT Powers
# from scipy import fftpack

# fft_powers = np.abs(fftpack.fft(lfps))
# fft_freqs = np.fft.fftfreq(lfps.shape[-1], d=1.0 / SAMP_RATE)
# mask = fft_freqs >= 0
# fft_powers, fft_freqs = fft_powers[:, mask], np.round(fft_freqs[mask], 1)
# fft_df = pd.DataFrame(data=fft_powers, columns=fft_freqs.astype(str))
fft_df = utils.dsp.calc_fft_powers(lfps, SAMP_RATE)
fft_freqs = fft_df.columns.astype(float)

## Calculates correlation coefficients between FFT powers and mean MEP magnitude
mean_mep = meps.mean(axis=-1)
# np.corrcoef(fft_df.iloc[:, 0], mean_mep)
take_corr_partial = partial(take_corr, y=mean_mep)
corr_coeffs = fft_df.apply(take_corr_partial)


## Statistical Test (no correlation test)
def r2t(r, dof):
    t = np.abs(np.array(r)) * np.sqrt(dof / (1 - np.array(r) ** 2))
    return t


dof = len(fft_df) - 2
ts = r2t(corr_coeffs, dof)

# t-values for the degree of freedom
t_005 = stats.t.ppf(1 - 0.05 / 2, dof)
t_001 = stats.t.ppf(1 - 0.01 / 2, dof)

# Check the relationship between the correlation coefficients (r) and t-values in a graph
r_axis = np.arange(-1, 1, 0.01)[1:-1]
t_axis = r2t(r_axis, dof)
plt.scatter(r_axis, t_axis)
plt.axhline(t_005, label="0.05 t-value", color="orange")
plt.axhline(t_001, label="0.01 t-value", color="red")
plt.xlabel("r value")
plt.ylabel("t value")
plt.title(
    "The relationship between r values (= the correlation coefficients) and t values (n={:,})".format(
        len(fft_df)
    )
)
plt.legend()
# plt.show()
utils.general.save(plt, SDIR + "mouse_#{}/r_and_t_.png".format(args.n_mouse))

# Finds not significant t-values and the FFT frequencies
ns_freqs_001 = fft_freqs.copy()[ts < t_001]
ns_freqs_005 = fft_freqs.copy()[ts < t_005]

ns_freq_001_txt = "\nN.S. freqs for alpha = .01: {}\n".format(ns_freqs_001)
ns_freq_005_txt = "\nN.S. freqs for alpha = .05: {}\n".format(ns_freqs_005)

print(ns_freq_005_txt)
print(ns_freq_001_txt)


## Plots the Figure
plt.plot(corr_coeffs)
plt.ylim(-1, 1)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Corr. coeff.")
plt.title(
    "Mouse#{nm} (n={n:,}); {ns_txt}".format(
        nm=args.n_mouse, n=len(fft_df), ns_txt=ns_freq_005_txt
    )
)
# plt.show()
utils.general.save(plt, SDIR + "mouse_#{}/corr_coeff.png".format(args.n_mouse))


## Saves
df = pd.DataFrame(
    {
        "freqs": fft_freqs,
        "rs": corr_coeffs,
        "n": np.ones_like(fft_freqs) * len(fft_df),
        "ts": ts,
        "are_significant_005": (t_005 <= ts),
        "are_significant_001": (t_001 <= ts),
    }
)

utils.general.save(df, SDIR + "mouse_#{nm}/corr_coeff.csv".format(nm=args.n_mouse))


## EOF
