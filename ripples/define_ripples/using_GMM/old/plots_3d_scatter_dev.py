#!/usr/bin/env python
import argparse
import sys

sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-s", "--save", default=False, choices=[False, "png", "mp4"], help=" ")
args = ap.parse_args()


## Functions
def adds_3d_scatter_subplot(
    fig,
    ax_position,
    cls0_sparse_df,
    ftr1,
    ftr2,
    ftr3,
    cls0_label=None,
    cls0_color_str="blue",
    cls1_sparse_df=None,
    cls1_label=None,
    cls1_color_str="red",
    title=None,
    spath_mp4=False,
    spath_png=False,
    theta=30,
    phi=30,
    size=10,
    xmin=2.5,
    xmax=8.0,
    ymin=-2.5,
    ymax=3.0,
    zmin=0.0,
    zmax=3.5,
    alpha=0.35,
):
    ##############################
    ## Parameters
    ##############################
    RGB_PALLETE_DICT = utils.general.load("./conf/global.yaml")["RGB_PALLETE_DICT"]

    ##############################
    ## Preparation
    ##############################
    # fig = plt.figure()
    ax = fig.add_subplot(*ax_position, projection="3d")
    ax.set_xlabel(ftr1)
    ax.set_ylabel(ftr2)
    ax.set_zlabel(ftr3)
    plt.title(title)
    ax.axis((2.5, 8.0, -2.5, 3.0))
    ax.set_zlim3d(bottom=0.0, top=3.5)
    ax.view_init(phi, theta)

    ##############################
    ## Plots
    ##############################
    ax.scatter(
        cls0_sparse_df[ftr1],
        cls0_sparse_df[ftr2],
        cls0_sparse_df[ftr3],
        marker="o",
        label=cls0_label,
        alpha=alpha,
        s=size,
        c=utils.plt.colors.to_RGBA(
            cls0_color_str,
            alpha=alpha,
        ),
    )

    if cls1_sparse_df is not None:
        ax.scatter(
            cls1_sparse_df[ftr1],
            cls1_sparse_df[ftr2],
            cls1_sparse_df[ftr3],
            marker="o",
            label=cls1_label,
            alpha=alpha,
            s=size,
            c=utils.plt.colors.to_RGBA(
                cls1_color_str,
                alpha=alpha,
            ),
        )

    plt.legend(loc="upper left")

    ax.axis((xmin, xmax, ymin, ymax))
    ax.set_zlim3d(bottom=zmin, top=zmax)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.zaxis.set_major_locator(MultipleLocator(1.0))

    ##############################
    ## Draws Cubes
    ##############################
    ax = utils.plt.draw_cube(
        [np.log(15), xmax],
        [ymin, np.log(1)],
        [np.log(2), zmax],
        c=utils.plt.colors.to_RGBA(
            "green",
            alpha=alpha,
        ),
        alpha=0.5,
    )

    r1, r2, r3 = [np.log(15), xmax], [ymin, np.log(1 * 5 / 4)], [np.log(4), zmax]
    ax = utils.plt.draw_cube(
        [np.log(15), xmax],
        [ymin, np.log(1 * 5 / 4)],
        [np.log(4), zmax],
        c=utils.plt.colors.to_RGBA(
            "purple",
            alpha=alpha,
        ),
        alpha=0.5,
    )
    return fig


## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Configure Matplotlib
utils.plt.configure_mpl(plt, figsize=(8.7, 10), legendfontsize="small")


## Fixes random seed
utils.general.fix_seeds(seed=42, np=np)


## FPATHs
args.n_mouse = "01"
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)


## Loads
rips_df_list = utils.pj.load.rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MOUSE,
    rip_sec_ver="GMM_labeled/D{}+".format(args.n_mouse),
)

rips_df = pd.concat(rips_df_list)
are_ripple_GMM = rips_df["are_ripple_GMM"]
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
T_GMM_sparse_rips_df = rips_df[are_ripple_GMM & indi_sparse]
F_GMM_sparse_rips_df = rips_df[~are_ripple_GMM & indi_sparse]


## Plots
spath_mp4 = (
    utils.general.mk_spath("videos/mouse#{}.mp4".format(args.n_mouse), makedirs=True)
    if args.save == "mp4"
    else None
)
spath_png = (
    utils.general.mk_spath("images/mouse#{}.png".format(args.n_mouse), makedirs=True)
    if args.save == "png"
    else None
)

utils.plt.configure_mpl(plt, figsize=(18.1, 10), legendfontsize="small", fontsize=7)
fig = plt.figure()
for i in range(6):
    fig = adds_3d_scatter_subplot(
        fig,
        (2, 3, i + 1),
        T_GMM_sparse_rips_df,
        ftr1,
        ftr2,
        ftr3,
        cls0_label="Cluster T",
        cls0_color_str="blue",
        cls1_sparse_df=F_GMM_sparse_rips_df,
        cls1_label="Cluster F",
        cls1_color_str="red",
        spath_png=spath_png,
        spath_mp4=spath_mp4,
        title="Mouse #{}\nSparsity: {}%\n".format(args.n_mouse, perc),
        theta=165,
        phi=3,
        size=10,
        alpha=0.35,
    )

utils.general.save(fig, "/tmp/test.mp4")


# fig, axes = plt.subplots(1, 2)
# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection="3d")
# ax1.plot(np.arange(10), np.arange(10), np.arange(10))
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.plot(np.arange(10), np.arange(10), np.arange(10))
# utils.general.save(fig, "/tmp/test.mp4")


## EOF
