#!/usr/bin/env python

import numpy as np
import utils


def lfps_rips_sec(fpaths_lfp, rip_sec_ver="candi_orig"):
    """ """
    # rip_sec_versions_list = ['candi_orig', 'candi_with_props', 'GMM_labeled', 'CNN_labeled']
    # assert rip_sec_ver in rip_sec_versions_list

    lfps, rips_sec = [], []
    for f in fpaths_lfp:
        f_rip = utils.pj.path_converters.LFP_to_ripples(f, rip_sec_ver=rip_sec_ver)
        lfps.append(np.load(f).squeeze())
        rips_sec.append(utils.general.load(f_rip))
    return lfps, rips_sec


def rips_sec(fpaths_lfp, rip_sec_ver="candi_orig"):
    rips_sec = []
    for f in fpaths_lfp:
        f_rip = utils.pj.path_converters.LFP_to_ripples(f, rip_sec_ver=rip_sec_ver)
        rips_sec.append(utils.general.load(f_rip))
    return rips_sec


def split_n_mice_tra_tes(i_mouse_test=0):
    """
    i_mouse_test = 0
    """
    N_MICE_CANDIDATES = ["01", "02", "03", "04", "05"]
    n_mice_tes = [N_MICE_CANDIDATES.pop(i_mouse_test)]
    n_mice_tra = N_MICE_CANDIDATES
    return n_mice_tra, n_mice_tes


def split_npy_list_tra_tes(n_mice_tra, n_mice_tes):
    LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
        "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
    )

    lpath_hippo_lfp_npy_list_tra = list(
        np.hstack(
            [
                utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, nm_tra)[1]
                for nm_tra in n_mice_tra
            ]
        )
    )

    lpath_hippo_lfp_npy_list_tes = list(
        np.hstack(
            [
                utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, nm_tes)[1]
                for nm_tes in n_mice_tes
            ]
        )
    )
    return lpath_hippo_lfp_npy_list_tra, lpath_hippo_lfp_npy_list_tes


def lfps_rips_tra_or_tes(
    # lpath_hippo_lfp_npy_list,
    tra_or_tes_str,
    i_mouse_test,
):

    # n_mice_tra, n_mice_tes = split_n_mice_tra_tes(i_mouse_test=i_mouse_test)
    lpath_hippo_lfp_npy_list_tra, lpath_hippo_lfp_npy_list_tes = split_npy_list_tra_tes(
        *split_n_mice_tra_tes(i_mouse_test=i_mouse_test)
        # n_mice_tra, n_mice_tes
    )

    if tra_or_tes_str == "tra":
        dataset_key = "D0{}-".format(str(i_mouse_test + 1))
        lpath_hippo_lfp_npy_list = lpath_hippo_lfp_npy_list_tra
    if tra_or_tes_str == "tes":
        dataset_key = "D0{}+".format(str(i_mouse_test + 1))
        lpath_hippo_lfp_npy_list = lpath_hippo_lfp_npy_list_tes

    lfps, rips = utils.pj.load.lfps_rips_sec(
        lpath_hippo_lfp_npy_list,
        rip_sec_ver="CNN_labeled/{}".format(dataset_key),
    )

    _rips_GMM = utils.pj.load.rips_sec(
        lpath_hippo_lfp_npy_list,
        rip_sec_ver="GMM_labeled/{}".format(dataset_key),
    )  # to take ln(ripple peak magni. / SD)

    ln_norm_ripple_peak_key = "ln(ripple peak magni. / SD)"
    for i_rip in range(len(rips)):
        rips[i_rip][ln_norm_ripple_peak_key] = _rips_GMM[i_rip][ln_norm_ripple_peak_key]
    del _rips_GMM

    return lfps, rips


# def load_mouse_lfp_rip(
#     mouse_num="01", dataset_plus_or_minus="+", rip_sec_ver="GMM_labeled"
# ):
#     LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
#         "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
#     )

#     ##############################
#     mouse_num = "01"
#     dataset_plus_or_minus = "+"
#     ##############################

#     lpath_hippo_lfp_npy_list_mouse = utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, mouse_num)[1]

#     dataset_key = "D" + mouse_num + dataset_plus_or_minus

#     lfps_mouse, rips_df_list = load_lfps_rips_sec(
#         lpath_hippo_lfp_npy_list_mouse, rip_sec_ver="{}/{}".format(rip_sec_ver, dataset_key)
#     )

#     return lfps_mouse, rips_df_list


# # def load_mouse_lfp_rip(
# #     mouse_num="01", dataset_plus_or_minus="+", rip_sec_ver="GMM_labeled"
# # ):
# #     LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
# #         "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
# #     )

# #     # mouse_num = "01"
# #     # dataset_plus_or_minus = "+"
# #     dataset_key = "D" + mouse_num + dataset_plus_or_minus
# #     lfp_fpaths_mouse = utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, mouse_num)[1]

# #     lfps_mouse, rips_df_list = load_lfps_rips_sec(
# #         lfp_fpaths_mouse, rip_sec_ver="{}/{}".format(rip_sec_ver, dataset_key)
# #     )  # includes labels using GMM on the dataset
# #     return lfps_mouse, rips_df_list


# # def pad_sequence(listed_1Darrays, padding_value=0):
# #     '''
# #     listed_1Darrays = rips_level_in_slices
# #     '''
# #     listed_1Darrays = listed_1Darrays.copy()
# #     dtype = listed_1Darrays[0].dtype
# #     # get max_len
# #     max_len = 0
# #     for i in range(len(listed_1Darrays)):
# #       max_len = max(max_len, len(listed_1Darrays[i]))
# #     # padding
# #     for i in range(len(listed_1Darrays)):
# #       # pad = (np.ones(max_len - len(listed_1Darrays[i])) * padding_value).astype(dtype)
# #       # listed_1Darrays[i] = np.concatenate([listed_1Darrays[i], pad])
# #       pad1 = int((max_len - len(listed_1Darrays[i])) / 2)
# #       pad2 = max_len - len(listed_1Darrays[i]) - pad1
# #       listed_1Darrays[i] = np.pad(listed_1Darrays[i], [pad1, pad2],
# #                                   'constant', constant_values=(padding_value))
# #     listed_1Darrays = np.array(listed_1Darrays)
# #     return listed_1Darrays
