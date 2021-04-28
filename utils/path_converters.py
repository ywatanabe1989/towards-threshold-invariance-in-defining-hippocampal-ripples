#!/usr/bin/env python

def LFP_to_ripple_candi_with_props(path_lfp):
    path_ripples = path_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                           .replace('/orig/', '/with_props/')\
                           .replace('.npy', '.pkl')
    return path_ripples

to_ripple_candi_with_props_lpath = LFP_to_ripple_candi_with_props


def LFP_to_MEP_magni(path_lfp):
    path_mep_magni_sd_normed = path_lfp.replace('orig', 'magni')\
                                       .replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
    return path_mep_magni_sd_normed


def LFP_to_ripple_magni(path_lfp):
    path_ripple_magni_sd_normed = path_lfp.replace('orig', 'magni')\
                                          .replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')
    return path_ripple_magni_sd_normed


