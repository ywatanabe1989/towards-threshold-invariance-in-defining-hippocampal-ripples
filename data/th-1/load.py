#!/usr/bin/env python3
# Time-stamp: "2021-09-07 13:54:23 (ylab)"

import utils
import mne
import glob
import numpy as np


################################################################################
### PATHs
################################################################################
# lpath_dat = "data/th-1/data/Mouse12-120806-raw/Mouse12-120806-01.dat"
lpath_eeg = "data/th-1/data/Mouse12-120806/Mouse12-120806.eeg"
lpath_xml = "data/th-1/data/Mouse12-120806/Mouse12-120806.xml"

################################################################################
## Loads
################################################################################
## Parses the binary text as 16-bit integers
lfp = np.fromfile(open(lpath_eeg, "r"), dtype=np.int16)
xml = utils.general.load(lpath_xml)

samp_rate = int(xml["fieldPotentials"]["lfpSamplingRate"])
print(xml["generalInfo"])  # Electrode group: 14 for the CA1
xml["anatomicalDescription"]["channelGroups"]["group"]

n_chs_all = 90

len_per_ch = len(lfp) / n_chs_all


def plot(chs_HIPP):
    lfps_HIPP = [lfp[ch_HIPP::n_chs_all] for ch_HIPP in chs_HIPP]  # ch_HIPP+1?
    dt_sec = 1.0 / samp_rate * 1000
    dt_ms = dt_sec * 1000
    len_tot_ms = len(lfps_HIPP[0]) / samp_rate * 1000
    time_ms = np.linspace(0, len_tot_ms - dt_ms, len(lfps_HIPP[0]))

    start_sec_plt = 0
    dur_sec_plt = 5
    end_sec_plt = start_sec_plt + dur_sec_plt

    x_plt = time_ms[samp_rate * start_sec_plt : samp_rate * end_sec_plt].squeeze()
    lfps_HIPP_plt = np.array(
        [
            l[samp_rate * start_sec_plt : samp_rate * end_sec_plt].squeeze()
            for l in lfps_HIPP
        ]
    )

    delta_amp = -750
    lfps_HIPP_plt += np.arange(len(lfps_HIPP_plt))[:, np.newaxis] * delta_amp

    plt.rcParams["figure.dpi"] = 1200

    fig, ax = plt.subplots()
    ax.plot(x_plt, lfps_HIPP_plt.T, linewidth=0.01)
    ax.set_title(chs_HIPP)
    fig.show()


chs_HIPP = np.array([64, 66, 68, 70, 72])
plot(chs_HIPP)
# plot(chs_HIPP+1)
