#!/usr/bin/env python

import argparse
import numpy as np
import sys
sys.path.append('.')
import myutils.mat2py as mat2py
import myutils.myfunc as mf


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mat", default='./data/01/day1/split/LFP_MEP_2kHz_mat/tt8-2.mat', \
                help="path to input .mat file")

ap.add_argument("-dtype", default=np.float16, \
                help=" ")
args = ap.parse_args()


## Parse File Paths
lpath = args.mat
ldir, fname, ext = mf.split_fpath(lpath)


## Load
data = mat2py.mat2npa(lpath, typ=args.dtype).squeeze()


## Save
if args.dtype == np.int16:
  dtype_txt = '_int16'
if args.dtype == np.float16:
  dtype_txt = '_fp16'

spath = ldir + fname + dtype_txt + ".npy"
np.save(spath, data)
print('Saved to {}'.format(spath))


## EOF
