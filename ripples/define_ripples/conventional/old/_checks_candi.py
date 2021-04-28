#!/usr/bin/env python

import os
import sys; sys.path.append('.')
from progs.utils.general import (load_pkl,
                                 save_pkl,
                                 split_fpath,
                                 )
from glob import glob


## Check
lpath_new_pkl = './data/01/day1/split/ripple_candi_1kHz_pkl/tt2-1_fp16.pkl'
ripple_candi_new_df = load_pkl(lpath_new_pkl)

lpath_old_pkl = '/mnt/10TB_RED/nvme_back/Ripple_Detection/data/01/day1/split/1kHz/tt2-1_fp16.pkl'
ripple_candi_old_df = load_pkl(lpath_new_pkl)

(ripple_candi_new_df == ripple_candi_old_df).all() # True



## Copy to reduce ripple candidate detecting time
lpaths_old_pkl = glob('/mnt/10TB_RED/nvme_back/Ripple_Detection/data/0?/day?/split/1kHz/tt?-?_rip_sec.pkl')


files = []
for f in lpaths_old_pkl:
    f = f.split('Ripple_Detection')[1]
    files.append(f.replace('1kHz', 'ripple_candi_1kHz_pkl').replace('_rip_sec', ''))


root_dir = '/mnt/nvme/Semisupervised_Ripples'
for f, lpath_old_pkl in zip(files, lpaths_old_pkl):
    spath = root_dir + f
    print(spath, lpath_old_pkl)
    sdir, _, _ = split_fpath(spath)
    os.makedirs(sdir, exist_ok=True)
    save_pkl(load_pkl(lpath_pkl), spath)

## EOF
