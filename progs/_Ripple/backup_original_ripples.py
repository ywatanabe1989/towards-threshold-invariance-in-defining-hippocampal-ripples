import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
import numpy as np
import myutils.myfunc as mf
from glob import glob
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Parameters
SAMP_RATE = 1000


## Parse File Paths
fpath_lfp = args.npy_fpath
fpath_ripples = fpath_lfp.replace('.npy', '_riptimes_sd1.pkl')


## Load
lfp = np.load(fpath_lfp).squeeze().astype(np.float32).squeeze()
ripples = mf.load_pkl(fpath_ripples)[['start_sec', 'end_sec']]


## Save
spath = fpath_lfp.replace('.npy', '_ripple_candi_100-250Hz.pkl')
mf.save_pkl(ripples, spath)


# ## EOF
