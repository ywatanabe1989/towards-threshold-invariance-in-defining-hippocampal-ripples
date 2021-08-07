#!/usr/bin/env python

import argparse
import sys

sys.path.append(".")
from itertools import combinations, product

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib import animation
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2


## Funcs
def draw_a_cube(ax, r1, r2, r3, c="blue", alpha=1.0):
    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0]:
            ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
        if np.sum(np.abs(s - e)) == r2[1] - r2[0]:
            ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
        if np.sum(np.abs(s - e)) == r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
    return ax


def plot_3d_scatter(
    cls0_sparse_df,
    ftr1,
    ftr2,
    ftr3,
    cls0_label=None,
    cls0_color_str="blue",
    cls1_sparse_df=None,
    cls1_label=None,
    cls1_color_str="red",
    cls2_sparse_df=None,
    cls2_label=None,
    cls2_color_str="yellow",
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
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
    # for sd, l, c in zip(
    #     [cls0_sparse_df, cls1_sparse_df, cls2_sparse_df],
    #     [cls0_label, cls1_label, cls2_label],
    #     [cls0_color_str, cls1_color_str, cls2_color_str],
    # ):
    #     try:
    #         ax.scatter(
    #             sd[ftr1],
    #             sd[ftr2],
    #             sd[ftr3],
    #             marker="o",
    #             label=l,
    #             alpha=alpha,
    #             s=size,
    #             c=utils.plt.colors.to_RGBA(
    #                 c,
    #                 alpha=alpha,
    #             ),
    #         )
    #     except:
    #         pass

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

    if cls2_sparse_df is not None:
        ax.scatter(
            cls2_sparse_df[ftr1],
            cls2_sparse_df[ftr2],
            cls2_sparse_df[ftr3],
            marker="o",
            label=cls2_label,
            alpha=alpha,
            s=size,
            c=utils.plt.colors.to_RGBA(
                cls2_color_str,
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
    ax = draw_a_cube(
        ax,
        [np.log(15), xmax],
        [ymin, np.log(1)],
        [np.log(2), zmax],
        c=utils.plt.colors.to_RGBA(
            "green",
            alpha=alpha,
        ),
        alpha=0.5,
    )

    ax = draw_a_cube(
        ax,
        [np.log(15), xmax],
        [ymin, np.log(1 * 5 / 4)],
        [np.log(4), zmax],
        c=utils.plt.colors.to_RGBA(
            "purple",
            alpha=alpha,
        ),
        alpha=0.5,
    )

    ##############################
    ## Saves
    ##############################
    if spath_png:  # as a Figure
        plt.savefig(spath_png)
        plt.close()
        print("\nSaved to: {}\n".format(spath_png))

    if spath_mp4:  # as a movie
        mk_mp4(fig, spath_mp4)

    ##############################
    ## or Just return the fig object
    ##############################
    else:
        return fig


def mk_mp4(fig, spath_mp4):
    axes = fig.get_axes()

    def init():
        return (fig,)

    def animate(i):
        for ax in axes:
            ax.view_init(elev=10.0, azim=i)
        # ax.view_init(elev=10.0, azim=i)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=360, interval=20, blit=True
    )

    writermp4 = animation.FFMpegWriter(fps=60, extra_args=["-vcodec", "libx264"])
    anim.save(spath_mp4, writer=writermp4)
    print("\nSaving to: {}\n".format(spath_mp4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "-nm",
        "--n_mouse",
        default="02",
        choices=["01", "02", "03", "04", "05"],
        help=" ",
    )
    args = ap.parse_args()

    ## Parse File Path
    LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
        "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
    )
    LPATHS_MOUSE = utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]

    ## Loads
    lfps, _rips_df = utils.pj.load.lfps_rips_sec(
        LPATHS_MOUSE, rip_sec_ver="ripple_candi_1kHz_pkl/with_props"
    )
    len_rips_df = np.array([len(_rip) for _rip in _rips_df])
    rips_df = pd.concat(_rips_df)

    # # Prepares sparse Data Frame for visualization
    # perc = .2 if args.n_mouse == '01' else .05
    # N = int(len(rips_df) * perc / 100)
    # indi_sparse = np.random.permutation(len(rips_df))[:N]
    # sparse_rips_df = rips_df.iloc[indi_sparse]

    ## Plots
    theta, phi = 165, 3
    ftr1, ftr2, ftr3 = (
        "ln(duration_ms)",
        "mean ln(MEP magni. / SD)",
        "ln(ripple peak magni. / SD)",
    )
    plot_3d_scatter(
        sparse_rips_df,
        cls0_label=None,
        spath_png=None,
        spath_mp4=None,
        theta=theta,
        phi=phi,
        perc=perc,
    )

    # # label_name_cleaned = 'label_cleaned_from_gmm_within_mouse{}'.format(args.n_mouse)
    # label_name_cleaned = 'label_cleaned_from_gmm_wo_mouse01'
    # indi_t_cleaned = sparse_rip_df[label_name_cleaned] == 0 # fixme
    # indi_f_cleaned = sparse_rip_df[label_name_cleaned] == 1
    # t_cleaned = sparse_rip_df[[ftr1, ftr2, ftr3]][indi_t_cleaned]
    # f_cleaned = sparse_rip_df[[ftr1, ftr2, ftr3]][indi_f_cleaned]

    # label_name_gmm = 'label_gmm'
    # indi_t_gmm = sparse_rip_df[label_name_gmm] == 0
    # indi_f_gmm = sparse_rip_df[label_name_gmm] == 1
    # t_gmm = sparse_rip_df[[ftr1, ftr2, ftr3]][indi_t_gmm]
    # f_gmm = sparse_rip_df[[ftr1, ftr2, ftr3]][indi_f_gmm]

    # theta, phi = 165, 3
    # # plot_3d_scatter(rips_df, t_gmm, f_gmm, plot_ellipsoid=False, theta=theta, phi=phi, perc=perc)
    # # plot_3d_scatter(rips_df, t_cleaned, f_cleaned, plot_ellipsoid=False, theta=theta, phi=phi, perc=perc)

    # indi_t2t = indi_t_gmm & indi_t_cleaned
    # indi_f2f = indi_f_gmm & indi_f_cleaned
    # indi_f2t = indi_f_gmm & indi_t_cleaned
    # indi_t2f = indi_t_gmm & indi_f_cleaned

    # t2f = sparse_rip_df[[ftr1, ftr2, ftr3]][indi_t2f]
    # f2t = sparse_rip_df[[ftr1, ftr2, ftr3]][indi_f2t]
    # plot_3d_scatter(rips_df, f2t, t2f, plot_ellipsoid=True, theta=theta, phi=phi, perc=perc)
    # # ## Save
    # # spath_root = '~/Desktop/fig3a/'
    # # pos_gmm.to_csv(spath_root + 'pos_gmm.csv')
    # # neg_gmm.to_csv(spath_root + 'neg_gmm.csv')
    # # pos_cleaned.to_csv(spath_root + 'pos_cleaned.csv')
    # # neg_cleaned.to_csv(spath_root + 'neg_cleaned.csv')
