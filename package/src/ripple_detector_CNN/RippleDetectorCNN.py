#!/usr/bin/env python3
# Time-stamp: "2021-09-18 08:04:04 (ywatanabe)"

import pkgutil
import random

import mngs
import numpy as np
import ripple_detector_CNN
import torch
import torch.nn as nn
import yaml


class RippleDetectorCNN(object):
    def __init__(
        self, lfp_1ch_cropped, lfp_timepoints_sec, window_size_pts=400, samp_rate=1000
    ):
        self.lfp_1ch_cropped = lfp_1ch_cropped
        self.lfp_timepoints_sec = lfp_timepoints_sec
        self.window_size_pts = window_size_pts
        self.samp_rate = samp_rate
        self.window_size_sec = window_size_pts / samp_rate

    def detect_ripple_candidates(self, lo_hz_ripple=150, hi_hz_ripple=250):
        """Detects ripple candidates with a lower biased method based on Kay et al."""
        lfp_1ch_cropped_h = len(self.lfp_1ch_cropped) / self.samp_rate / 60 / 60
        print(f"\nDetecting ripple candidates (Length: {lfp_1ch_cropped_h:.1f}h)\n")

        _, _, rip_sec_df = ripple_detector_CNN.define_ripple_candidates(
            self.lfp_timepoints_sec,
            self.lfp_1ch_cropped,
            self.samp_rate,
            lo_hz=150,
            hi_hz=250,
            zscore_threshold=1,
        )

        rip_sec_df = rip_sec_df.reset_index()
        rip_sec_df["duration_sec"] = rip_sec_df["end_sec"] - rip_sec_df["start_sec"]

        self.rip_sec_df = rip_sec_df.copy()  # to the attribute

        return rip_sec_df

    def estimate_ripple_proba(self, batch_size=32, checkpoints=None):
        self.rip_sec_df = self._check_if_ripple_candidates_are_separable(
            self.rip_sec_df
        )
        self.rip_sec_df = self._separate_ripple_candidates(self.rip_sec_df)
        self._load_the_trained_CNN(checkpoints=checkpoints)
        softmax = nn.Softmax(dim=-1)
        # labels = {0: "F", 1: "T"}

        self.rip_sec_df["ripple_conf"] = np.nan  # initialization

        n_batches = len(self.rip_sec_df) // batch_size + 1
        for i_batch in range(n_batches):
            ## Samples
            start = i_batch * batch_size
            end = (i_batch + 1) * batch_size
            _Xb = self.rip_sec_df["cut_LFP"].iloc[start:end]
            indi_isolated = _Xb.index[~_Xb.isna()]
            Xb = self.rip_sec_df["cut_LFP"].iloc[indi_isolated]

            ## NN Outputs
            Xb = torch.tensor(np.vstack(Xb).astype(np.float32)).cuda()
            logits = self.model(Xb)
            pred_proba = softmax(logits)
            # pred_class = pred_proba.argmax(dim=-1)  # 0: T, 1: F

            self.rip_sec_df.loc[indi_isolated, "ripple_conf"] = (
                pred_proba[:, 1].detach().cpu().numpy().copy()
            )
        return self.rip_sec_df.copy()

    def _check_if_ripple_candidates_are_separable(self, rip_sec_df):
        """Checks whether each ripple can be separated in a segment"""

        def _check_if_a_ripple_candidate_is_separable(
            rip_sec_df, i_rip, window_size_sec
        ):
            ## Condition 1
            rip_i = rip_sec_df.iloc[i_rip]
            is_rip_i_short = (rip_i["end_sec"] - rip_i["start_sec"]) < window_size_sec

            ## Condition 2
            if i_rip != 0:
                rip_i_minus_1 = rip_sec_df.iloc[i_rip - 1]
                before_end_sec = rip_i_minus_1["end_sec"]
            else:
                before_end_sec = 0

            if i_rip != len(rip_sec_df) - 1:
                rip_i_plus_1 = rip_sec_df.iloc[i_rip + 1]
                next_start_sec = rip_i_plus_1["start_sec"]
            else:
                next_start_sec = before_end_sec + window_size_sec

            is_inter_ripple_interval_long = window_size_sec < (
                next_start_sec - before_end_sec
            )
            is_rip_i_separable = is_rip_i_short & is_inter_ripple_interval_long
            return is_rip_i_separable

        rip_sec_df["is_separable"] = [
            _check_if_a_ripple_candidate_is_separable(
                rip_sec_df, i_rip, self.window_size_sec
            )
            for i_rip in range(len(rip_sec_df))
        ]

        return rip_sec_df

    def _separate_ripple_candidates(
        self,
        rip_sec_df,
    ):
        def _separate_a_ripple_candidate(
            lfp, rip_sec_df, i_rip, window_size_pts=400, samp_rate=1000
        ):
            """Isolates each sesparable ripple with random starting point"""
            rip_sec_df_i = rip_sec_df.iloc[i_rip].copy()
            dur_sec = rip_sec_df_i["end_sec"] - rip_sec_df_i["start_sec"]
            dur_pts = int(dur_sec * samp_rate)
            dof = window_size_pts - dur_pts
            cut_start_pts = (
                int(rip_sec_df_i["start_sec"] * samp_rate)
                - random.randint(0, dof)
                - int(self.lfp_timepoints_sec[0] * samp_rate)
            )
            lfp_cut = lfp[cut_start_pts : cut_start_pts + window_size_pts]  # fixme
            return lfp_cut

        rip_sec_df["cut_LFP"] = np.nan
        rip_sec_df["cut_LFP"] = rip_sec_df["cut_LFP"].astype(object)
        for i_rip in range(len(rip_sec_df)):
            if rip_sec_df.iloc[i_rip]["is_separable"]:
                rip_sec_df.loc[i_rip, "cut_LFP"] = _separate_a_ripple_candidate(
                    self.lfp_1ch_cropped,
                    rip_sec_df,
                    i_rip,
                    window_size_pts=self.window_size_pts,
                    samp_rate=self.samp_rate,
                )

                if len(rip_sec_df.loc[i_rip, "cut_LFP"]) != self.window_size_pts:
                    rip_sec_df.loc[i_rip, "cut_LFP"] = np.nan
                    rip_sec_df.loc[i_rip, "is_separable"] = False

        return rip_sec_df

    def _load_the_trained_CNN(self, checkpoints=None):
        ## Model initialization

        ResNet1D_conf = next(
            yaml.load_all(pkgutil.get_data("ripple_detector_CNN", "data/ResNet1D.yaml"))
        )
        model = ripple_detector_CNN.ResNet1D(ResNet1D_conf)

        ## Loads the trained weight on four mice's data
        if checkpoints is None:
            # checkpoints = mngs.general.load(
            #     mngs.general.get_data_path_from_a_package(
            #         "./ripples/detect_ripples/CNN/train_FvsT/checkpoints/mouse_test#01_epoch#000.pth"
            #     )
            # )
            checkpoints = mngs.general.load(
                "./ripples/detect_ripples/CNN/train_FvsT/checkpoints/mouse_test#01_epoch#000.pth"
            )
        model.load_state_dict(
            mngs.general.cvt_multi2single_model_state_dict(
                checkpoints["model_state_dict"]
            )
        )

        ## to GPU and evaluation mode
        self.model = model.to("cuda").eval()
