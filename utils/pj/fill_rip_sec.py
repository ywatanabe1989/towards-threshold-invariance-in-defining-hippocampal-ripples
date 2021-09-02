#!/usr/bin/env python

import numpy as np
import pandas as pd


def fill_rip_sec(lfp, rip_sec, samp_rate):
    rip_sec["is_candidate"] = True

    rip_sec_level_0 = pd.DataFrame()
    rip_sec_level_0["end_sec"] = rip_sec["start_sec"] - 1 / samp_rate
    rip_sec_level_0["start_sec"] = (
        np.hstack((np.array(0) - 1 / samp_rate, rip_sec["end_sec"][:-1].values))
        + 1 / samp_rate
    )
    cols = rip_sec_level_0.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    rip_sec_level_0 = rip_sec_level_0[cols]
    # add last row
    lfp_end_sec = len(lfp) / samp_rate
    last_row_dict = {
        "start_sec": [rip_sec.iloc[-1]["end_sec"]],
        "end_sec": [lfp_end_sec],
    }
    last_row_index = rip_sec_level_0.index[-1:] + 1
    last_row_df = pd.DataFrame(data=last_row_dict).set_index(last_row_index)
    rip_sec_level_0 = rip_sec_level_0.append(last_row_df)

    keys = rip_sec.keys()
    filled_ripples = pd.concat([rip_sec, rip_sec_level_0], sort=False)[keys]
    filled_ripples = filled_ripples.sort_values("start_sec")
    # filled_ripples['is_candidate'] = filled_ripples["is_candidate"].fillna("False")  # filled_ripples
    # filled_ripples['is_candidate'] = filled_ripples['is_candidate'].astype(bool)

    filled_ripples["is_candidate"] = filled_ripples["is_candidate"].fillna(False)

    return filled_ripples
