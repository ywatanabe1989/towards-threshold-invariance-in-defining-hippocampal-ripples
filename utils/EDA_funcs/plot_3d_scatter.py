#!/usr/bin/env python
import argparse
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
from scipy.stats import chi2

import utils.general as ug
import utils.semi_ripple as us
import ffmpeg


## Funcs
def plot_3d_scatter(cls0_sparse_df,
                    ftr1,
                    ftr2,
                    ftr3,
                    cls0_label=None,
                    cls0_color_str='blue',
                    cls1_sparse_df=None,
                    cls1_label=None,
                    cls1_color_str='red',                    
                    title=None,
                    spath_mp4=False,
                    spath_png=False,
                    theta=30,
                    phi=30,
                    size=10,
                    xmin=2.5,
                    xmax=8.,
                    ymin=-2.5,
                    ymax=3.,
                    zmin=0.,
                    zmax=3.5,
                    alpha=.3,
                    ):
    
    ##############################
    ## Parameters
    ##############################        
    RGB_PALLETE_DICT = ug.load_yaml_as_dict('./conf/global.yaml')['RGB_PALLETE_DICT']

    ##############################
    ## Preparation
    ##############################        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(ftr1)
    ax.set_ylabel(ftr2)
    ax.set_zlabel(ftr3)
    plt.title(title)
    ax.axis((2.5, 8., -2.5, 3.))
    ax.set_zlim3d(bottom=0., top=3.5)
    ax.view_init(phi, theta)

    ##############################
    ## Plots
    ##############################        
    ax.scatter(cls0_sparse_df[ftr1], cls0_sparse_df[ftr2], cls0_sparse_df[ftr3],
               marker='o', label=cls0_label, alpha=alpha, s=size,
               c=(np.array(RGB_PALLETE_DICT[cls0_color_str]) / 255)[np.newaxis, ...])

    if cls1_sparse_df is not None:
        ax.scatter(cls1_sparse_df[ftr1], cls1_sparse_df[ftr2], cls1_sparse_df[ftr3],
                   marker='o', label=cls1_label, alpha=alpha, s=size,
                   c=(np.array(RGB_PALLETE_DICT[cls1_color_str]) / 255)[np.newaxis, ...])

    plt.legend(loc='upper left')

    ax.axis((xmin, xmax, ymin, ymax))
    ax.set_zlim3d(bottom=zmin, top=zmax)
    ax.yaxis.set_major_locator(MultipleLocator(1.))
    ax.zaxis.set_major_locator(MultipleLocator(1.))

    ##############################
    ## Draws Cubes
    ##############################    
    def draw_cube(r1, r2, r3, c='blue', alpha=1.):
        from itertools import product, combinations
        for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
            if np.sum(np.abs(s-e)) == r1[1]-r1[0]:
                ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
            if np.sum(np.abs(s-e)) == r2[1]-r2[0]:
                ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
            if np.sum(np.abs(s-e)) == r3[1]-r3[0]:
                ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
    

    r1, r2, r3 = [np.log(15), xmax], [ymin, np.log(1)], [np.log(2), zmax]
    draw_cube(r1, r2, r3,
              c=tuple(np.array(RGB_PALLETE_DICT['green']) / 255),
              alpha=alpha,
              )
  
    r1, r2, r3 = [np.log(15), xmax], [ymin, np.log(1*5/4)], [np.log(4), zmax]
    draw_cube(r1, r2, r3,
              c=tuple(np.array(RGB_PALLETE_DICT['purple']) / 255),
              alpha=alpha,
              )

    ##############################
    ## Saves
    ##############################    
    if spath_png: # as a Figure
        plt.savefig(spath_png)
        plt.close()
        print("\nSaved to: {}\n".format(spath_png))

    if spath_mp4: # as a movie
        def init():
            return fig,

        def animate(i):
            ax.view_init(elev=10., azim=i)
            return fig,
      
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       init_func=init,
                                       frames=360,
                                       interval=20,
                                       blit=True)
        
        writermp4 = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
        anim.save(spath_mp4, writer=writermp4)
        print('\nSaving to: {}\n'.format(spath_mp4))

    ##############################
    ## Just plots
    ##############################    
    else:
        plt.show()

      



if __name__ == '__main__':
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-nm", "--n_mouse", default='02', choices=['01', '02', '03', '04', '05'], \
                    help=" ")
    args = ap.parse_args()
    
    ## Parse File Path
    LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
    LPATHS_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]
    
    ## Loads
    lfps, _rips_df = us.load_lfps_rips_sec(LPATHS_MOUSE,
                                           rip_sec_ver='ripple_candi_1kHz_pkl/with_props'
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
    ftr1, ftr2, ftr3 = 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)'
    plot_3d_scatter(sparse_rips_df, cls0_label=None, spath_png=None, spath_mp4=None,
                    theta=theta, phi=phi, perc=perc)



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
