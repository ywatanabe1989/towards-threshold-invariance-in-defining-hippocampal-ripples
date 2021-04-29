#!/usr/bin/env python
import numpy as np
import utils.general as ug


def load_lfps_rips_sec(fpaths_lfp, rip_sec_ver='candi_orig'):
    '''
    '''
    rip_sec_versions_list = ['candi_orig', 'candi_with_props', 'GMM_labeled', 'CNN_labeled']
    assert rip_sec_ver in rip_sec_versions_list
    
    lfps, rips_sec = [], []
    for f in fpaths_lfp:
        f_rip = f.replace('LFP_MEP_1kHz_npy', 'ripples_1kHz_pkl')\
                 .replace('/orig/', '/{}/'.format(rip_sec_ver))\
                 .replace('.npy', '.pkl')
        lfps.append(np.load(f).squeeze())
        rips_sec.append(ug.load_pkl(f_rip))
    return lfps, rips_sec



# def pad_sequence(listed_1Darrays, padding_value=0):
#     '''
#     listed_1Darrays = rips_level_in_slices
#     '''
#     listed_1Darrays = listed_1Darrays.copy()
#     dtype = listed_1Darrays[0].dtype
#     # get max_len
#     max_len = 0
#     for i in range(len(listed_1Darrays)):
#       max_len = max(max_len, len(listed_1Darrays[i]))
#     # padding
#     for i in range(len(listed_1Darrays)):
#       # pad = (np.ones(max_len - len(listed_1Darrays[i])) * padding_value).astype(dtype)
#       # listed_1Darrays[i] = np.concatenate([listed_1Darrays[i], pad])
#       pad1 = int((max_len - len(listed_1Darrays[i])) / 2)
#       pad2 = max_len - len(listed_1Darrays[i]) - pad1
#       listed_1Darrays[i] = np.pad(listed_1Darrays[i], [pad1, pad2],
#                                   'constant', constant_values=(padding_value))
#     listed_1Darrays = np.array(listed_1Darrays)
#     return listed_1Darrays
