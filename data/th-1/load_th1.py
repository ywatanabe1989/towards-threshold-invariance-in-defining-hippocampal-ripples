#!/usr/bin/env python3
# Time-stamp: "2021-09-17 20:32:02 (ywatanabe)"

import numpy as np
import mngs


def load_th1(
    lpath_eeg="./data/th-1/data/Mouse12-120806/Mouse12-120806.eeg",
    start_sec_cut=0,
    dur_sec_cut=5,
):

    lpath_xml = lpath_eeg.replace(".eeg", ".xml")

    ################################################################################
    ## Loads
    ################################################################################
    lfp = np.fromfile(open(lpath_eeg, "r"), dtype=np.int16)
    xml = mngs.general.load(lpath_xml)

    ################################################################################
    ## Gets parameters
    ################################################################################
    samp_rate = int(xml["fieldPotentials"]["lfpSamplingRate"])
    # print(xml["generalInfo"])  # Electrode group: 14 for the CA1
    xml["anatomicalDescription"]["channelGroups"]["group"]

    n_chs_all = 90

    len_per_ch = len(lfp) / n_chs_all

    chs_HIPP = np.array([64, 66, 68, 70, 72])
    lfps_HIPP = np.array(
        [lfp[ch_HIPP::n_chs_all] for ch_HIPP in chs_HIPP]
    )  # ch_HIPP+1?

    dt_sec = 1.0 / samp_rate * 1000
    dt_ms = dt_sec * 1000
    len_tot_ms = len(lfps_HIPP[0]) / samp_rate * 1000
    time_ms = np.linspace(0, len_tot_ms - dt_ms, len(lfps_HIPP[0]))

    end_sec_cut = start_sec_cut + dur_sec_cut

    if dur_sec_cut != -1:
        x_cut = time_ms[samp_rate * start_sec_cut : samp_rate * end_sec_cut].squeeze()
    else:
        x_cut = time_ms[samp_rate * start_sec_cut :].squeeze()
    lfps_HIPP_cut = np.array(
        [
            l[samp_rate * start_sec_cut : samp_rate * end_sec_cut].squeeze()
            for l in lfps_HIPP
        ]
    )

    prop_dict = dict(
        samp_rate=samp_rate,
        chs_HIPP=chs_HIPP,
    )

    return lfps_HIPP_cut, prop_dict
