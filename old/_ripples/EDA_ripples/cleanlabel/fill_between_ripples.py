import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
import numpy as np
import myutils.myfunc as mf
import pandas as pd


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Funcs
def fill_undefined_rip_sec(lfp, rip_sec, samp_rate):
    rip_sec_level_0 = pd.DataFrame()
    rip_sec_level_0['end_sec'] = rip_sec['start_sec'] - 1/samp_rate
    rip_sec_level_0['start_sec'] = np.hstack((np.array(0) - 1/samp_rate, rip_sec['end_sec'][:-1].values)) + 1/samp_rate
    cols = rip_sec_level_0.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    rip_sec_level_0 = rip_sec_level_0[cols]
    # add last row
    lfp_end_sec = len(lfp) / samp_rate
    last_row_dict = {'start_sec':[rip_sec.iloc[-1]['end_sec']],
                     'end_sec':[lfp_end_sec],
                     }
    last_row_index = rip_sec_level_0.index[-1:] + 1
    last_row_df = pd.DataFrame(data=last_row_dict).set_index(last_row_index)
    rip_sec_level_0 = rip_sec_level_0.append(last_row_df)
    return rip_sec_level_0


## Parameters
SAMP_RATE = 1000


## Parse File Paths
fpath_lfp = args.npy_fpath
ldir, fname, ext = mf.split_fpath(fpath_lfp)
fpath_ripples = fpath_lfp.replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_merged.pkl')

## Load
lfp = np.load(fpath_lfp).squeeze().astype(np.float32)[:, np.newaxis]
ripples = mf.load_pkl(fpath_ripples)


## Fill undefined times as False Ripple time
fillers = fill_undefined_rip_sec(lfp, ripples, SAMP_RATE)
keys = ripples.keys()
filled_ripples = pd.concat([ripples,
                              fillers],
                              sort=False)[keys]
filled_ripples = filled_ripples.sort_values('start_sec')


## Save
spath = fpath_ripples.replace('.pkl', '_filled.pkl')
mf.save_pkl(filled_ripples, spath)


## EOF
