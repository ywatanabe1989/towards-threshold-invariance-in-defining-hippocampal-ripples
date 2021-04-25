#!/usr/bin/env python

def to_ripple_candi_with_props_lpath(lpath_lfp):
    lpath_ripples = lpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                             .replace('/orig/', '/with_props/')\
                             .replace('.npy', '.pkl')
    return lpath_ripples


