#!/usr/bin/env python

'''
This file does the followings:
    1) To load a *.mat file
    2) To down-sample the loaded 1d LFP (local field potential) or MEP (myoelectric potential) data.
    3) To save as numpy file (.npy) with designated data type (default: fp16)
'''

import numpy as np    

## Function
def load_a_prepared_mat_file_as_an_arr(lpath_mat, dtype=np.float32):
    '''Load .mat file as a numpy array.
       The difference between the outputs' format of nsx2mat_matlab.m(hdf5 dataset) and
       nsx2mat_octave.m (matlab 5.0 MAT-file, by Octave 5.2.0) are absorbed in this function.
    
    Example:
        # pseudo-output of nsx2mat_matlab.m
        lpath_matlab = './data/01/day1/split_matlab/LFP_MEP_2kHz_mat/tt8-2.mat'

        # pseudo-output of nsx2mat_octave.m
        lpath_octave = './data/01/day1/split_octave/LFP_MEP_2kHz_mat/tt8-2.mat'
    
        data_mat = load_a_prepared_mat_file_as_an_arr(lpath_matlab, dtype=args.dtype)
        data_oct = load_a_prepared_mat_file_as_an_arr(lpath_octave, dtype=args.dtype)
        print((data_mat == data_oct).mean()) # 1.0
    '''
  
    import h5py
    from scipy.io import loadmat

    
    is_hdf5 = h5py.is_hdf5(lpath_mat)
    if is_hdf5:
        f = h5py.File(lpath_mat)      
    else:
        f = loadmat(lpath_mat)

    arrs_dict = {}
    for k, v in f.items():
        arrs_dict[k] = np.array(v)

    data_1d = (arrs_dict['save_data']).squeeze().astype(dtype)
    return data_1d
  
  
if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from scipy.signal import decimate

    import sys; sys.path.append('.')
    from utils.general import (split_fpath,
                               to_str_dtype,
                               to_str_samp_rate,
                               get_samp_rate_str_from_fpath,
                               to_int_samp_rate,
                               )


    ## Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--lpath_mat",
                    default='./data/okada/01/day1/split/LFP_MEP_2kHz_mat/tt8-2.mat', \
                    help="path to input .mat file")
    ap.add_argument("--dtype", default='np.float16', type=str, \
                    help=" ")
    ap.add_argument("-ts", "--tgt_samp_rate", default=1000, type=int, \
                    help=" ")
    args = ap.parse_args()
    args.dtype = eval(args.dtype)


    ## PATHs
    ldir, fname, ext = split_fpath(args.lpath_mat)

    
    ## Load
    data_2kHz = load_a_prepared_mat_file_as_an_arr(args.lpath_mat, dtype=np.float32) # (183421162,)


    ## Down samplingg
    SRC_SAMP_RATE = to_int_samp_rate(get_samp_rate_str_from_fpath(args.lpath_mat)) # 2000
    TGT_SAMP_RATE = args.tgt_samp_rate # 1000
    DOWN_SAMP_FACTOR = int(SRC_SAMP_RATE/TGT_SAMP_RATE) # 2
    data_xHz = decimate(data_2kHz, DOWN_SAMP_FACTOR).astype(args.dtype) # (91710581,)
    
    
    ## Save
    dtype_str = to_str_dtype(args.dtype)
    tgt_samp_rate_str = to_str_samp_rate(args.tgt_samp_rate)    
    sdir = ldir.replace('LFP_MEP_2kHz_mat', 'LFP_MEP_{}_npy/orig'.format(tgt_samp_rate_str))
    os.makedirs(sdir, exist_ok=True)    
    spath = sdir + fname + '_' + dtype_str + ".npy"
    # './data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt8-2_fp16.npy'
    np.save(spath, data_xHz)
    print('Saved to: {}'.format(spath))


    ## EOF
