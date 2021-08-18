#!/usr/bin/env python

import numpy as np
import utils
from tqdm import tqdm


def rip_sec(lpath_lfp, rip_sec_ver="candi_orig", cycle_dataset=False, n_mouse=None):

    lpath_rip = utils.pj.path_converters.LFP_to_ripples(
        lpath_lfp, rip_sec_ver=rip_sec_ver
    )

    if cycle_dataset:  # +1
        lpath_rip = utils.pj.path_converters.cycle_dataset(lpath_rip, n_mouse)

    ## Loads
    rip_sec = utils.general.load(lpath_rip)

    if "CNN_labeled" in rip_sec_ver:
        rip_sec = utils.pj.invert_ripple_labels(rip_sec)
        rip_sec = utils.pj.add_ripple_props(lpath_lfp, rip_sec)

    return rip_sec


def rips_sec(lpaths_lfp, rip_sec_ver="candi_orig", cycle_dataset=False, n_mouse=None):
    rips_sec = []

    print("\nLoading ...\n")
    for lpath_lfp in tqdm(lpaths_lfp):
        rip_sec = utils.pj.load.rip_sec(
            lpath_lfp,
            rip_sec_ver=rip_sec_ver,
            cycle_dataset=cycle_dataset,
            n_mouse=n_mouse,
        )
        rips_sec.append(rip_sec)
    return rips_sec


def lfps_rips_sec(
    lpaths_lfp, rip_sec_ver="candi_orig", cycle_dataset=False, n_mouse=None
):
    """
    # rip_sec_versions_list = ['candi_orig', 'candi_with_props', 'GMM_labeled', 'CNN_labeled']
    # assert rip_sec_ver in rip_sec_versions_list
    """

    print("\nLoading ...\n")
    lfps, rips_sec = [], []
    for lpath_lfp in tqdm(lpaths_lfp):
        ## Loads
        lfp = np.load(lpath_lfp).squeeze()
        rip_sec = utils.pj.load.rip_sec(
            lpath_lfp,
            rip_sec_ver=rip_sec_ver,
            cycle_dataset=cycle_dataset,
            n_mouse=n_mouse,
        )

        ## Buffering
        lfps.append(lfp)
        rips_sec.append(rip_sec)

    return lfps, rips_sec


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
    tra_or_tes_str,
    i_mouse_test,
    is_debug=False,
):

    lpath_hippo_lfp_npy_list_tra, lpath_hippo_lfp_npy_list_tes = split_npy_list_tra_tes(
        *split_n_mice_tra_tes(i_mouse_test=i_mouse_test)
    )

    if tra_or_tes_str == "tra":
        dataset_key = "D0{}-".format(str(i_mouse_test + 1))
        lpath_hippo_lfp_npy_list = lpath_hippo_lfp_npy_list_tra

    if tra_or_tes_str == "tes":
        dataset_key = "D0{}+".format(str(i_mouse_test + 1))
        lpath_hippo_lfp_npy_list = lpath_hippo_lfp_npy_list_tes

    if is_debug:  # a shortcut for debugging
        # N_LOAD_FOR_DEBUG = 2
        N_LOAD_FOR_DEBUG = -1
        if tra_or_tes_str == "tra":
            tra_or_tes_str_for_print = "training"
        if tra_or_tes_str == "tes":
            tra_or_tes_str_for_print = "test"
        print(
            "\n!!! DEBUG MODE !!!\nOnly {} files are loaded as the {} dataset.\n".format(
                N_LOAD_FOR_DEBUG,
                tra_or_tes_str_for_print,
            )
        )
        lpath_hippo_lfp_npy_list = lpath_hippo_lfp_npy_list[:N_LOAD_FOR_DEBUG]

    lfps, rips = utils.pj.load.lfps_rips_sec(
        lpath_hippo_lfp_npy_list,
        rip_sec_ver="CNN_labeled/{}".format(dataset_key),
    )

    return lfps, rips


def get_hipp_lfp_fpaths(mouse_ids):
    if not isinstance(mouse_ids, list):
        mouse_ids = [mouse_ids]

    HIPP_LFP_PATHS_NPY_ALL = utils.general.load(
        "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
    )

    HIPP_LFP_PATHS_NPY_MICE = [
        utils.general.grep(HIPP_LFP_PATHS_NPY_ALL, nm)[1] for nm in mouse_ids
    ]

    HIPP_LFP_PATHS_NPY_MICE = list(np.hstack(HIPP_LFP_PATHS_NPY_MICE))

    return HIPP_LFP_PATHS_NPY_MICE
