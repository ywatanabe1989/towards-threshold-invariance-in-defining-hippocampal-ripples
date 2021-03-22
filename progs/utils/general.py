#!/usr/bin/env python

import numpy as np
import re
import time

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


def to_int_samp_rate(samp_rate_int):
    TO_INT_SAMP_RATE_DICT = {'2kHz':2000, '1kHz':1000, '500kHz': 500}
    return TO_INT_SAMP_RATE_DICT[samp_rate_int]
    
def to_str_samp_rate(samp_rate_str):
    TO_STR_SAMP_RATE_DICT = {2000: '2kHz', 1000: '1kHz', 500: '500kHz'}
    return TO_STR_SAMP_RATE_DICT[samp_rate_str]
    

def get_samp_rate_str_from_fpath(fpath):
    samp_rate_candi_str = ['2kHz', '1kHz', '500Hz']
    for samp_rate_str in samp_rate_candi_str:
        matched = re.search(samp_rate_str, fpath)
        is_matched = not (matched is None)
        if is_matched:
            return samp_rate_str


def calc_h(data, sampling_rate):
    return len(data) / sampling_rate / 60 / 60


def save_pkl(obj, fpath):
    import pickle
    with open(fpath, 'wb') as f: # 'w'
        pickle.dump(obj, f)
    print('Saved to: {}'.format(fpath))
    

def save_npy(np_arr, fpath):
    np.save(fpath, np_arr)
    print('Saved to: {}'.format(fpath))

def load_pkl(fpath, print=False):
    obj = pkl_load(fpath)
    if print:
        print('Loaded: {}'.format(fpath))    
    return obj

def load_npy(fpath, print=False):
    arr = np.load(fpath)
    if print:    
        print('Loaded: {}'.format(fpath))
    return arr

class time_tracker():
    def __init__(self):
        self.id = -1
        self.start = time.time()
        self.prev = self.start

    def __call__(self, comment=None):
        now = time.time()
        from_start = now - self.start
        self.from_start_hhmmss = time.strftime('%H:%M:%S', time.gmtime(from_start))
        from_prev = now - self.prev
        self.from_prev_hhmmss = time.strftime('%H:%M:%S', time.gmtime(from_prev))
        self.id += 1
        self.prev = now
        if comment:
            print("Time (id:{}): tot {}, prev {} [hh:mm:ss]: {}\n".format(\
                  self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment))
        else:
            print("Time (id:{}): tot {}, prev {} [hh:mm:ss]\n".format(\
                  self.id, self.from_start_hhmmss, self.from_prev_hhmmss))
