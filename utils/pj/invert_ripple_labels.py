#!/usr/bin/env python

import numpy as np


def invert_ripple_labels(_rips_df):
    rips_df = _rips_df.copy()
    ## checks if already inverted or not
    has_errors = rips_df["are_errors"].sum() == len(rips_df)
    if not has_errors:
        _already_inverted = rips_df["are_errors"].copy()

        ## Prob.
        rips_df["pred_probas_ripple_CNN"] = rips_df["psx_ripple"].copy()  # initializes
        pred_probas_ripple_CNN = np.array(
            1 - rips_df["psx_ripple"][rips_df["are_errors"]].copy()
        )  # inverts
        rips_df.loc[
            rips_df["are_errors"], "pred_probas_ripple_CNN"
        ] = pred_probas_ripple_CNN.copy()

        ## Label
        rips_df["are_ripple_CNN"] = rips_df["are_ripple_GMM"].copy()  # initializes
        rips_df.loc[rips_df["are_errors"], "are_ripple_CNN"] = ~rips_df[
            "are_ripple_GMM"
        ][rips_df["are_errors"]]

        rips_df["already_inverted"] = _already_inverted

    return rips_df
