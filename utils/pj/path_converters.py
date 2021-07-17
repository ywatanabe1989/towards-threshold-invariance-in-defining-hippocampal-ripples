#!/usr/bin/env python

from itertools import cycle

import numpy as np
import utils


def LFP_to_ripples(fpath_lfp, rip_sec_ver="candi_orig"):
    candi_vers = ["candi_orig", "candi_with_props", "isolated"]
    GMM_vers = list(
        np.hstack(
            [
                ["GMM_labeled/D0{}+".format(i + 1), "GMM_labeled/D0{}-".format(i + 1)]
                for i in range(5)
            ]
        )
    )
    CNN_vers = list(
        np.hstack(
            [
                ["CNN_labeled/D0{}+".format(i + 1), "CNN_labeled/D0{}-".format(i + 1)]
                for i in range(5)
            ]
        )
    )

    rip_sec_vers = candi_vers + GMM_vers + CNN_vers

    assert rip_sec_ver in rip_sec_vers

    fpath_rip = (
        fpath_lfp.replace("LFP_MEP_1kHz_npy", "ripples_1kHz_pkl")
        .replace("/orig/", "/{}/".format(rip_sec_ver))
        .replace(".npy", ".pkl")
    )
    return fpath_rip


def LFP_to_MEP_magni(path_lfp):
    path_mep_magni_sd_normed = path_lfp.replace("orig", "magni").replace(
        "_fp16.npy", "_mep_magni_sd_fp16.npy"
    )
    return path_mep_magni_sd_normed


def LFP_to_ripple_magni(path_lfp):
    path_ripple_magni_sd_normed = path_lfp.replace("orig", "magni").replace(
        "_fp16.npy", "_ripple_band_magni_sd_fp16.npy"
    )
    return path_ripple_magni_sd_normed


def LFP_to_MEPs(lpath_lfp):
    _LDIR, _, _ = utils.general.split_fpath(lpath_lfp)
    _FPATHS_TRAPE_MEP = utils.general.load(
        "./data/okada/FPATH_LISTS/TRAPE_MEP_TT_NPYs.txt"
    )
    LPATHs_MEP = utils.general.grep(_FPATHS_TRAPE_MEP, _LDIR)[1]
    return LPATHs_MEP


def cycle_dataset(lpath_including_Dxxx, n_mouse):
    n_mice_all = ["0{}".format(i + 1) for i in range(5)]
    cycle_n_mice_all = cycle(n_mice_all)
    for i_nm, nm in enumerate(cycle_n_mice_all):
        if n_mouse == nm:
            tgt_n_mouse = next(cycle_n_mice_all)
            return lpath_including_Dxxx.replace(
                "D{}".format(n_mouse), "D{}".format(tgt_n_mouse)
            )
