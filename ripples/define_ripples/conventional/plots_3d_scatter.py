#!/usr/bin/env python

import argparse
import sys

sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
ap.add_argument("-s", "--save", default=False, choices=[False, "png", "mp4"], help=" ")
args = ap.parse_args()


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys)


## Configure Matplotlib
utils.plt.configure_mpl(plt, legendfontsize="small")


## Fixes random seed
utils.general.fix_seeds(seed=42, np=np)


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = utils.general.grep(
    LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse
)[1]


## Loads
lfps, rips_df_list = utils.pj.load.lfps_rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MOUSE, rip_sec_ver="candi_with_props"
)
rips_df = pd.concat(rips_df_list)
ftr1, ftr2, ftr3 = (
    "ln(duration_ms)",
    "ln(mean MEP magni. / SD)",
    "ln(ripple peak magni. / SD)",
)
rips_df = rips_df[[ftr1, ftr2, ftr3]]


## Prepares sparse Data Frame for visualization
perc = 0.20 if args.n_mouse == "01" else 0.05
N = int(len(rips_df) * perc / 100)
_indi_sparse = np.random.permutation(len(rips_df))[:N]
indi_sparse = np.zeros(len(rips_df))
indi_sparse[_indi_sparse] = 1
indi_sparse = indi_sparse.astype(bool)


## Defines clusters
cls0_sparse_rips_df = rips_df[indi_sparse]
# print(cls0_sparse_rips_df.iloc[:10])


## Plots
spath_mp4 = (
    utils.general.mk_spath("mouse_{n}.mp4".format(n=args.n_mouse), makedirs=True)
    if args.save == "mp4"
    else None
)
spath_png = (
    utils.general.mk_spath("mouse_{n}.png".format(n=args.n_mouse), makedirs=True)
    if args.save == "png"
    else None
)
utils.pj.plot_3d_scatter(
    cls0_sparse_rips_df,
    ftr1,
    ftr2,
    ftr3,
    cls0_label="Ripple Candidates",
    cls0_color_str="gray",
    spath_png=spath_png,
    spath_mp4=spath_mp4,
    title="Mouse #{}\nSparsity: {}%\n".format(args.n_mouse, perc),
    theta=165,
    phi=3,
    size=10,
    alpha=0.5,
)


## EOF
