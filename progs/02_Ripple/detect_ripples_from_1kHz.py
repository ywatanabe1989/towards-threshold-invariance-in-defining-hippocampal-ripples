import argparse
import numpy as np
# from ripple_detection import Kay_ripple_detector
import numpy as np
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
from detectors import detect_ripple_candi
import myutils.myfunc as mf

mytime = mf.time_tracker()

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Parameters
samp_rate = 1000
# sd_thresh = args.sd_thresh


## Load
fpath = args.npy_fpath
lfp = np.load(fpath).squeeze().astype(np.float32)
lfp = lfp[:, np.newaxis] # The shape of LFP should be (len(lfp), 1) to fullfil the requirement of the ripple detector.

start_sec, end_sec, step_sec = 0, 1.*len(lfp)/samp_rate, 1.0/samp_rate
time_x = np.arange(start_sec, end_sec, step_sec)
lfp = lfp[int(start_sec*samp_rate):int(end_sec*samp_rate)]


## Detect Ripple Candidates
print('Detecting ripples from {} (Length: {:.1f}h)'.format(fpath, len(lfp)/samp_rate/3600))
rip_sec = detect_ripple_candi(time_x, lfp, samp_rate, lo_hz=100, hi_hz=250, zscore_threshold=1)
mytime()


## Save
savedir, fname, ext = mf.split_fpath(fpath)
savepath = savedir + fname + '_ripples.pkl' # .format(sd_thresh)
mf.pkl_save(rip_sec, savepath)


## EOF
