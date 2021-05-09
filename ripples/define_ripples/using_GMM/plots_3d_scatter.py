#!/usr/bin/env python
import argparse
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.general as ug
import utils.semi_ripple as us
from utils.EDA_funcs.plot_3d_scatter import plot_3d_scatter



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
ap.add_argument("-s", "--save", default=False, choices=[False, 'png', 'mp4'], \
                help=" ")
args = ap.parse_args()


## Configure Matplotlib
ug.configure_mpl(plt)


## Fixes random seed
ug.fix_seeds(seed=42, np=np)


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]


## Loads
lfps, rips_df_list = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MOUSE,
                                           rip_sec_ver='GMM_labeled/D{}+'.format(args.n_mouse)
                                           )
# len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]
rips_df = pd.concat(rips_df_list)
are_ripple_GMM = rips_df['are_ripple_GMM']
ftr1, ftr2, ftr3 = 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)'
rips_df = rips_df[[ftr1, ftr2, ftr3]]


## Prepares sparse Data Frame for visualization
perc = .20 if args.n_mouse == '01' else .05
N = int(len(rips_df) * perc / 100)
_indi_sparse = np.random.permutation(len(rips_df))[:N]
indi_sparse = np.zeros(len(rips_df))
indi_sparse[_indi_sparse] = 1
indi_sparse = indi_sparse.astype(bool)


## Defines clusters
T_GMM_sparse_rips_df = rips_df[are_ripple_GMM & indi_sparse]
F_GMM_sparse_rips_df = rips_df[~are_ripple_GMM & indi_sparse]


## Plots
spath_mp4 = ug.mk_spath('mouse_{n}.mp4'.format(n=args.n_mouse), makedirs=True) \
            if args.save == 'mp4' else None
spath_png = ug.mk_spath('mouse_{n}.png'.format(n=args.n_mouse), makedirs=True) \
            if args.save == 'png' else None    
plot_3d_scatter(T_GMM_sparse_rips_df,
                ftr1,
                ftr2,
                ftr3,
                cls0_label='Cluster T',
                cls0_color_str='blue',
                cls1_sparse_df=F_GMM_sparse_rips_df,
                cls1_label='Cluster F',
                cls1_color_str='red',                
                spath_png=spath_png,
                spath_mp4=spath_mp4,
                title='Mouse #{}\nSparsity: {}%\n'.format(args.n_mouse, perc),
                theta=165,
                phi=3,
                size=10,                
                alpha=.5,
                )

## EOF
