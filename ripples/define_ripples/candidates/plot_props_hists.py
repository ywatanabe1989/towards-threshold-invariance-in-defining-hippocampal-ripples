#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils.general as ug
import utils.path_changers as up


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--num_mouse",
                default='01', \
                help=" ")
args = ap.parse_args()


## PATHs
hipp_lfp_paths_npy = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
hipp_lfp_paths_npy_mouse_i = ug.search_str_list(hipp_lfp_paths_npy, args.num_mouse)[1]


## Loads
rip_sec_df = [ug.load_pkl(up.to_ripple_candi_with_props_lpath(f))
              for f in hipp_lfp_paths_npy_mouse_i]
rip_sec_df = pd.concat(rip_sec_df)


## Takes natural log of duration_ms
rip_sec_df['ln(duration_ms)'] = np.log(rip_sec_df['duration_ms'])

## Excludes columns
rip_sec_df = rip_sec_df[['ln(duration_ms)',
                         'mean ln(MEP magni. / SD)',
                         'ln(ripple peak magni. / SD)']]

## Plots
plot = False
if plot:
    n_bins = 500
    fig, ax = plt.subplots(1,3)
    plt.title('Mouse#{}'.format(args.num_mouse))
    ax[0].hist(rip_sec_df['ln(duration_ms)'], bins=n_bins, label='ln(Duration) [a.u.]')
    ax[0].set_ylim(0, 60000)
    ax[0].set_xlim(3, 9)
    ax[0].legend()

    ax[1].hist(rip_sec_df['mean ln(MEP magni. / SD)'], bins=n_bins,
               label='ln(Mean normalized magnitude of MEP) [a.u.]')
    ax[1].set_ylim(0, 40000)

    ax[1].set_xlim(-3, 3)
    ax[1].legend()


    ax[2].hist(rip_sec_df['ln(ripple peak magni. / SD)'], bins=n_bins,
               label='ln(Normalized ripple peak magnitude) [a.u.]')
    ax[2].set_ylim(0, 40000)
    ax[2].set_xlim(-1, 3)
    ax[2].legend()

    plt.show()


## Saves
## Saves only 10% of all samples
perc = 10
indi_random_10_perc = ug.get_random_indi(rip_sec_df, perc=perc)
rip_sec_df = rip_sec_df.iloc[indi_random_10_perc]
ug.save(rip_sec_df, 'rip_sec_df_mouse_{n}_{p}_perc.csv'\
        .format(n=args.num_mouse, p=perc))




## EOF
