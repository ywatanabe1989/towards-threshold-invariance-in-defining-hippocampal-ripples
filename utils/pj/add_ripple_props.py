#!/usr/bin/env python

import utils


def add_ripple_props(fpath_lfp, _rip_df):
    rip_sec = _rip_df.copy()

    FTR1, FTR2, FTR3 = (
        "ln(duration_ms)",
        "ln(mean MEP magni. / SD)",
        "ln(ripple peak magni. / SD)",
    )

    rip_sec_with_prop = utils.pj.load.rip_sec(fpath_lfp, rip_sec_ver="candi_with_props")

    rip_sec[FTR1] = rip_sec_with_prop[FTR1]
    rip_sec[FTR2] = rip_sec_with_prop[FTR2]
    rip_sec[FTR3] = rip_sec_with_prop[FTR3]

    return rip_sec
