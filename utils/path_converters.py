#!/usr/bin/env python


def LFP_to_ripples(fpath_lfp, rip_sec_ver='candi_orig'):
    rip_sec_versions_list = ['candi_orig', 'candi_with_props', 'GMM_labeled', 'CNN_labeled']
    assert rip_sec_ver in rip_sec_versions_list
    
    fpath_rip = fpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripples_1kHz_pkl')\
                         .replace('/orig/', '/{}/'.format(rip_sec_ver))\
                         .replace('.npy', '.pkl')
    return fpath_rip



# def LFP_to_ripple_candi_with_props(path_lfp):
#     path_ripples = path_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
#                            .replace('/orig/', '/with_props/')\
#                            .replace('.npy', '.pkl')
#     return path_ripples

# to_ripple_candi_with_props_lpath = LFP_to_ripple_candi_with_props


def LFP_to_MEP_magni(path_lfp):
    path_mep_magni_sd_normed = path_lfp.replace('orig', 'magni')\
                                       .replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
    return path_mep_magni_sd_normed


def LFP_to_ripple_magni(path_lfp):
    path_ripple_magni_sd_normed = path_lfp.replace('orig', 'magni')\
                                          .replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')
    return path_ripple_magni_sd_normed


