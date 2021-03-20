#!/usr/bin/env python

import numpy as np

def split_fpath(fpath):
    '''Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
    Example:
        dirname, fname, ext = split_fpath('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
        print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
        print(fname) # 'tt8-2'
        print(ext) # '.mat'
    '''
    import os
    dirname = os.path.dirname(fpath) + '/'
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext


def to_str_dtype(dtype):
    if dtype == np.int16:
        return 'int16'
    elif dtype == np.float16:
        return 'fp16'
    else:
        return None


def to_str_samp_rate(samp_rate_int):
    if samp_rate_int == 2000:
        return '2kHz'
    if samp_rate_int == 1000:
        return '1kHz'
    if samp_rate_int == 500:
        return '500Hz'
