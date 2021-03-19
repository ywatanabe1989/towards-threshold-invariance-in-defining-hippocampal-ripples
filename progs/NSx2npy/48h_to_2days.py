#!/usr/bin/env python

import argparse
import numpy as np
import h5py
import sys
# sys.path.append('..')
sys.path.append('.')
import myutils.mat2py
import myutils.myfunc as mf
from tqdm import tqdm


# mkdir ../data/05/day5/split -pv
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--numpy", default='../data/05/day4/split/2kHz/1_tt5-3_fp16.npy', help="path to input .npy file")
args = ap.parse_args()


## Func
def calc_h(data, sampling_rate):
    return len(data) / sampling_rate / 60 / 60


## Load
data1 = np.load(args.numpy)
data2 = np.load(args.numpy.replace('/1_', '/2_'))
data = np.concatenate((data1, data2), axis=0)


## Confirming lenghts
SAMP_RATE = 2000
data1_h = calc_h(data1, SAMP_RATE) # len(data1) / SAMP_RATE / 60 / 60
data2_h = calc_h(data2, SAMP_RATE)
tot_h = data1_h + data2_h
tot_h == len(data) / SAMP_RATE / 60 / 60

## Saving
data_1st_24h = data[:24*60*60*SAMP_RATE]
spath_1 = args.numpy.replace('/1_', '/')
np.save(spath_1, data_1st_24h)
print('Saved to: {}'.format(spath_1))

data_2nd_24h = data[24*60*60*SAMP_RATE:]
spath_2 = spath_1.replace('day4', 'day5')
np.save(spath_2, data_2nd_24h)
print('Saved to {}'.format(spath_2))
