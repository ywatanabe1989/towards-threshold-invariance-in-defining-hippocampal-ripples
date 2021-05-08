#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

import sys; sys.path.append('.')
from ripples.define_ripples.conventional.defines_ripple_candidates import define_ripple_candidates
import utils.general as ug
import utils.path_converters as upcvt
import utils.semi_ripple as us


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()




################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATH_HIPPO_LFP_NPY_LIST_MICE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]

lpath_lfp = LPATH_HIPPO_LFP_NPY_LIST_MICE[0]
lpath_rip_sec = upcvt.LFP_to_ripples(lpath_lfp,
                                     rip_sec_ver='CNN_labeled/D{}+'.format(args.n_mouse),
                                     )
lpath_rip_magni = upcvt.LFP_to_ripple_magni(lpath_lfp)
lpath_mep_magni = upcvt.LFP_to_MEP_magni(lpath_lfp)

## Gets Parameters
samp_rate = ug.get_samp_rate_int_from_fpath(lpath_lfp)
dt_sec = 1. / samp_rate


################################################################################
## Loads
################################################################################
lfp = ug.load(lpath_lfp)
rip_sec = ug.load(lpath_rip_sec)
rip_magni = ug.load(lpath_rip_magni)
mep_magni = ug.load(lpath_mep_magni)



## Makes ripple analog signal
rip_analog_sig = np.zeros_like(lfp)
rip_pred_proba_sig = np.zeros_like(lfp)
for i_rip, rip in rip_sec.iterrows():
    start_pts = int(rip['start_sec']*samp_rate)
    end_pts = int(rip['end_sec']*samp_rate)
    rip_analog_sig[start_pts:end_pts] = 1
    rip_pred_proba_sig[start_pts:end_pts] = rip['pred_probas_ripple_CNN']


################################################################################
## Plots
################################################################################
def plot_traces(start_sec=6516):
    # start_sec = 6516 # 6514 # 6514 # F2T, T2F
    end_sec = start_sec + 3 # 5 # 3880
    x_sec = np.arange(start_sec, end_sec, dt_sec) # x

    start_pts = int(start_sec*samp_rate)
    end_pts = int(end_sec*samp_rate)

    lfp_plt = lfp[start_pts:end_pts].squeeze()
    rip_analog_sig_plt = rip_analog_sig[start_pts:end_pts].squeeze()
    rip_magni_plt = rip_magni[start_pts:end_pts].squeeze()
    mep_magni_plt = mep_magni[start_pts:end_pts].squeeze()
    rip_pred_proba_sig_plt = rip_pred_proba_sig[start_pts:end_pts].squeeze()

    ## Gets ripple band LFP
    RIPPLE_CANDI_LIM_HZ = ug.load('./conf/global.yaml')['RIPPLE_CANDI_LIM_HZ']
    filted_plt, _, _ = define_ripple_candidates(x_sec,
                                                               lfp_plt,
                                                               samp_rate,
                                                               lo_hz=RIPPLE_CANDI_LIM_HZ[0],
                                                               hi_hz=RIPPLE_CANDI_LIM_HZ[1],
                                                               ) # 150, 250

    # Plot
    linewidth = 1
    fig, ax = plt.subplots(6, 1, sharex=True)
    ax[0].plot(x_sec, rip_pred_proba_sig_plt, linewidth=linewidth,
               label='estimated ripple probability')
    ax[1].plot(x_sec, lfp_plt, linewidth=linewidth,
               label='raw LFP')
    rip_band_str = '{}-{} Hz'.format(RIPPLE_CANDI_LIM_HZ[0], RIPPLE_CANDI_LIM_HZ[1])
    ax[2].plot(x_sec, filted_plt, linewidth=linewidth,
               label='ripple band LFP ()'.format(rip_band_str))
    ax[3].plot(x_sec, rip_magni_plt, linewidth=linewidth,
               label='ripple band normalized magnitude')
    ax[4].plot(x_sec, mep_magni_plt, linewidth=linewidth,
               label='MEP normalized magnitude')
    ax[5].plot(x_sec, rip_analog_sig_plt, linewidth=linewidth,
               label='ripple periods')

    for i in range(len(ax)):
        ax[i].legend()
    fig.show()

# s = np.random.randint()
plot_traces()





# # Limits
# ax[0].set_ylim(-1000, 1000)
# ax[1].set_ylim(-500, 500)
# ax[2].set_ylim(0, 300)
# # ax[4].set_ylim(0, 16)

# ## Ripple Coloring
# rip_sec_plt = rip_sec[(start_sec < rip_sec['start_sec']) & (rip_sec['end_sec'] < end_sec )]
# for i in range(4):
#     put_legend = False
#     first = True
#     for ripple in rip_sec_plt.itertuples():
#         ## Flip
#         conf = ripple.prob_pred_by_ResNet_wo_mouse02

#         if (ripple.label_gmm == 0) & (ripple.label_cleaned_from_gmm_wo_mouse02 == 0):
#           flip_txt = 'T2T \n{:.2f}'.format(conf)
#         if (ripple.label_gmm == 0) & (ripple.label_cleaned_from_gmm_wo_mouse02 == 1):
#           flip_txt = 'T2F \n{:.2f}'.format(1-conf)
#         if (ripple.label_gmm == 1) & (ripple.label_cleaned_from_gmm_wo_mouse02 == 0):
#           flip_txt = 'F2T \n{:.2f}'.format(conf)
#         if (ripple.label_gmm == 1) & (ripple.label_cleaned_from_gmm_wo_mouse02 == 1):
#           flip_txt = 'T2F \n{:.2f}'.format(1-conf)

#         ax[0].text((ripple.start_sec+ripple.end_sec)/2, 800, flip_txt)

#         if first:
#             label = 'Ripple Candi.' if put_legend else None
#             ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1, color='red', zorder=1000, label=label)
#             first = False
#         else:
#             ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1, color='red', zorder=1000)


## EOF    
