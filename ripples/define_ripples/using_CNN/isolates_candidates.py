#!/usr/bin/env python3

# import argparse
# import sys; sys.path.append('.')
# from rippledetection.core import filter_ripple_band, gaussian_smooth
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch.utils.data as utils

#!/usr/bin/env python
import argparse
from sklearn.mixture import GaussianMixture
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.general as ug
import utils.semi_ripple as us
import utils.path_converters as upcvt



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()


## Configure Matplotlib
ug.configure_mpl(plt)


## Fixes random seed
ug.fix_seed(seed=42)


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]


## Loads
lfps, rips_df_list = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MOUSE,
                                       rip_sec_ver='with_props'
                                       )
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]
rips_df = pd.concat(rips_df_list)




## Isolates Cut ripple candidates and make synthetic dataset
cut_lfps = []
max_duration_pts = 400
for i_lfp in range(len(lfps)):
    lfp = lfps[i_lfp]
    ripple = _ripples[i_lfp]
    for i_ripple in range(len(ripple)):
        start_pts = math.floor(ripple.loc[i_ripple+1, 'start_sec']*SAMP_RATE)
        end_pts = math.ceil(ripple.loc[i_ripple+1, 'end_sec']*SAMP_RATE)
        duration_pts = end_pts - start_pts
        center_pts = int((start_pts + end_pts) / 2)
        peak_pos_pts = np.clip(int(ripple.loc[i_ripple+1, 'ripple_peaks_pos_sec']*SAMP_RATE), start_pts, end_pts)
        assert start_pts <= peak_pos_pts & peak_pos_pts <= end_pts

        peak_to_start_pts = peak_pos_pts - start_pts
        peak_to_end_pts = abs(peak_pos_pts - end_pts)

        ## Centerize the peak position
        if peak_to_start_pts <= peak_to_end_pts:
            pad1 = abs(peak_to_end_pts - peak_to_start_pts)
            pad2 = 0
        if peak_to_end_pts < peak_to_start_pts:
            pad1 = 0
            pad2 = abs(peak_to_end_pts - peak_to_start_pts)

        cut_lfp = np.pad(lfp[start_pts:end_pts], [pad1, pad2], 'constant', constant_values=(0))

        if max_duration_pts < len(cut_lfp):
            cut_lfp = cut_lfp[int(len(cut_lfp)/2)-200:int(len(cut_lfp)/2)+200]

        cut_lfps.append(cut_lfp)

assert  len(cut_lfps) == len(ripples)

synthesized_ripples = pad_sequence(cut_lfps) # sythetic data
# mf.save_npy(synthesized_ripples, '../data/{}/synthesized_ripples_peak_centered.npy'.format(args.n_mouse))
