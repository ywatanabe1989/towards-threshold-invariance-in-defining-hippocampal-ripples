#!/usr/bin/env python

import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import skimage
import torch
import utils
from natsort import natsorted
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset


## Functions
class DataloaderFiller:
    """

    i_test_mouse:
        The index of test mouse

    window_size:
       The time length [points] to perform windowing
       (= epoching, binning, or segmenting) EEG signals.

    do_under_sampling:
        True enables under sampling with regard to diagnosed classes
        on training and validation dataset.

    dtype:
        'fp32' (default) or 'fp16'
    """

    def __init__(
        self,
        i_test_mouse=0,
        window_size=400,
        batch_size=64,
        num_workers=0,
        do_under_sampling=True,
        dtype="fp16",
        RANDOM_STATE=42,
        **kwargs,
    ):

        ################################################################################
        ## Fix random seed
        ################################################################################
        self.RANDOM_STATE = RANDOM_STATE
        utils.general.fix_seeds(seed=RANDOM_STATE, random=random, np=np, torch=torch)

        ################################################################################
        ## Attributes
        ################################################################################
        self.i_test_mouse = i_test_mouse
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.do_under_sampling = do_under_sampling
        self.dtype_np = np.float32 if dtype == "fp32" else np.float16
        self.dtype_torch = torch.float32 if dtype == "fp32" else torch.float16

        ################################################################################
        ## Load dataset
        ################################################################################

        # self.data_all = v4.treat_analysis_periods.load_diags_pkl(
        #     nomontage_or_IA_str="IA",
        #     include_endN_to_startNplus1=use_noisy_periods,
        #     load_diags_str=list(load_diags_str_expanded),
        # ).reset_index()

        ################################################################################
        ## Gets dataset information
        ################################################################################
        # # The labels for diagnosis are tagged as the same order the names were passed.
        # self.diag_str_to_int_dict = {k: i for i, k in enumerate(load_diags_str_conc)}
        # print("\nLabels were set as follows:\n{}.\n".format(self.diag_str_to_int_dict))
        # self.subj_ids_uq = natsorted(np.unique(self.data_all["Subject"]))
        # self.id_allocator_subj_global = utils.general.IDAllocator(self.subj_ids_uq)
        # self.diag_subj_uq_sets = self.data_all[
        #     ["Diagnosis", "Subject"]
        # ].drop_duplicates()

        ################################################################################
        ## A partial function
        ################################################################################
        # self.subj_to_diag_partial = partial(
        #     self._subj_to_diag, diag_subj_uq_sets=self.diag_subj_uq_sets
        # )

        ################################################################################
        ## Data splitting
        ################################################################################
        self._first_under_sampled = False

        # self.n_subj_in_training_dataset = len(self.subj_ids_tra)
        # self.n_subj_in_validation_dataset = len(self.subj_ids_val)
        # self.n_subj_in_test_dataset = len(self.subj_ids_tes)

        ################################################################################
        ## Make each mouse's EEG into segments
        ################################################################################
        self._viewed_data_all_dict = self._window_from_each_subj(
            self.data_all,
            window_size_pts=window_size,
            max_n_EEGs_per_subj=max_n_EEGs_per_subj,
        )  # fixme; using torch

        ################################################################################
        ## Fills dataloader
        ################################################################################
        self.not_filled_yet = True
        self.fill()

    def fill(
        self,
    ):  # fixme; random start time should be the same across mice for speed.
        """Everytime this method is called, training data loader is refilled.
        This enables to reduce the sampling bias.
        Validation and test dataset are filled only once, at the first time.
        """
        self.samples_tra, self.dl_tra = self._pack(
            self.subj_ids_tra, self.id_allocator_subj_tra, do_shuffle=True
        )

        # Only at the start of the first epoch
        if self.not_filled_yet:
            # validation dataset
            self.samples_val, self.dl_val = self._pack(
                self.subj_ids_val, self.id_allocator_subj_val, do_shuffle=False
            )
            self._samples_val, self._dl_val = (
                deepcopy(self.samples_val),
                deepcopy(self.dl_val),
            )

            # test dataset
            self.samples_tes, self.dl_tes = self._pack(
                self.subj_ids_tes, self.id_allocator_subj_tes, do_shuffle=False
            )
            self._samples_tes, self._dl_tes = (
                deepcopy(self.samples_tes),
                deepcopy(self.dl_tes),
            )

            self.not_filled_yet = False

        else:
            # After the 2nd epoch, validation and test dataloaders are
            # copied from the master copy.
            self.dl_val = deepcopy(self._dl_val)
            self.dl_tes = deepcopy(self._dl_tes)

    # def _train_valid_test_split(self, i_fold):
    #     """Under the "self.NUM_FOLDS"-fold CV setting, this function split all subject IDs into
    #     NUM_FOLDS * (training, validation, and test subdatasets) and take the i_fold's subdatasets.
    #     """
    #     skf = StratifiedKFold(
    #         n_splits=self.NUM_FOLDS, shuffle=True, random_state=self.RANDOM_STATE
    #     )
    #     X = np.array(self.diag_subj_uq_sets["Subject"])
    #     y = np.array(self.diag_subj_uq_sets["Diagnosis"])

    #     cv_d = utils.general.listed_dict(["Training", "Valid", "Test"])

    #     # y is passed to balance between classes. see StratifiedKFold.
    #     for train_index, test_index in skf.split(X, y):
    #         train_index, valid_index = train_test_split(
    #             train_index, test_size=self.VAL_RATIO
    #         )

    #         cv_d["Training"].append(train_index)
    #         cv_d["Valid"].append(valid_index)
    #         cv_d["Test"].append(test_index)

    #     for k in cv_d.keys():
    #         cv_d[k] = cv_d[k][i_fold]  # extract indices in the designated fold
    #         cv_d[k] = X[cv_d[k]]  # index to subjects

    #     if self.do_under_sampling:
    #         cv_d["Training"] = self._under_sample(cv_d["Training"], "Training")
    #         cv_d["Valid"] = self._under_sample(cv_d["Valid"], "Validation")

    #     return [cv_d[k] for k in ["Training", "Valid", "Test"]]

    # def _under_sample(self, subj_ids, step_key): # fixme
    #     """Perform under-sampling within a training dataset or a validation dataset.
    #     This enables for the dataset to have the same number of classes.
    #     """
    #     diags_str = np.array(
    #         [self.subj_to_diag_partial(subj_str) for subj_str in subj_ids]
    #     )  # from subject ID to diagnosis
    #     diag_uq = np.unique(diags_str)

    #     Ns_before = pd.Series(
    #         {"{}".format(ds): (diags_str == ds).sum() for ds in diag_uq}
    #     )

    #     N_min = min(Ns_before)

    #     Ns_after = pd.Series({"{}".format(ds): N_min for ds in diag_uq})

    #     print(
    #         "\nUnder-sampling on {} dataset:\nSample sizes\n\nBefore:\n{}\n\nAfter:\n{}".format(
    #             step_key,
    #             Ns_before,
    #             Ns_after,
    #         )
    #     )

    #     _indi_classes = [np.where(diags_str == ds)[0] for ds in diag_uq]
    #     _indi_pick = np.hstack([ic[:N_min] for ic in _indi_classes])
    #     subj_ids = subj_ids[_indi_pick]

    #     if self._first_under_sampled:
    #         print_txt = "\nBy under-sampling, the following subjects were selected as \
    #               {step_key} dataset \n{subj_keys}\n".format(
    #             step_key=step_key,
    #             subj_keys=subj_ids,
    #         )
    #         print(utils.general.squeeze_spaces(print_txt))
    #         # self._first_under_sampled = False

    #     return subj_ids

    def _pack(
        self,
        subj_list,
        subj_id_allocator_in_the_subdataset,
        do_shuffle=False,
        drop_last=True,
    ):
        '''Return "arrs_list_to_pack" and "dataloader"'''
        arrs_list_to__pack = self._sample_from_a_subdataset(
            self._viewed_data_all_dict,
            subj_list,
            subj_id_allocator_in_the_subdataset,
            do_shuffle=do_shuffle,
        )

        dl = DataLoader(
            TensorDataset(*[torch.tensor(d) for d in arrs_list_to__pack]),
            batch_size=self.batch_size,
            shuffle=do_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return arrs_list_to__pack, dl

    def _sample_from_a_subdataset(
        self,
        viewed_data_all_dict,
        subj_in_the_subdataset,
        subj_id_allocator_in_the_subdataset,
        do_shuffle=False,
    ):
        """Formats EEG and Meta Data in Training, Valid, or Test Dataset."""

        X, T, S_str = [], [], []
        for subj_str in subj_in_the_subdataset:
            _X = viewed_data_all_dict[subj_str]  # sample n*EEG segments
            X.append(_X)
            T.append(
                [
                    self.diag_str_to_int_dict[self.subj_to_diag_partial(subj_str)]
                    for _ in range(len(_X))
                ]
            )

            S_str.append([subj_str for _ in range(len(_X))])
        X, T, S_str = np.vstack(X), np.hstack(T), np.hstack(S_str)

        S = self.id_allocator_subj_global(S_str)  ## str to int
        S_reallocated = subj_id_allocator_in_the_subdataset(S_str)

        if do_shuffle == True:
            X, T, S, S_reallocated = shuffle(X, T, S, S_reallocated)

        X = X.transpose(0, 2, 1)  # to [bs, n_chs, siq_len]

        ## dtype
        X = X.astype(self.dtype_np)

        return X, T, S, S_reallocated

    def _window_from_each_subj(
        self, v4_df, window_size_pts=400, max_n_EEGs_per_subj=100
    ):

        d = utils.general.listed_dict(self.subj_ids_uq)
        for subj_str in self.subj_ids_uq:
            EEG_samples, is_sampled = self._sample_from_a_subj(
                v4_df, subj_str, window_size=window_size_pts, max_n_EEGs_per_subj=100
            )

            if is_sampled.all():
                d[subj_str] = EEG_samples

        return d

    def _sample_from_a_subj(
        self, v4_df, subj_str, window_size=400, max_n_EEGs_per_subj=100
    ):
        sub_df = v4_df[v4_df["Subject"] == subj_str]

        # init value is randomly chosen per subj per epoch
        rand_init_value = random.randint(0, window_size)
        viewed = sub_df["EEG"].apply(
            self._view_an_EEG_session,
            init_value=rand_init_value,
            window_size=window_size,
            n_chs=19,
        )
        viewed = np.vstack(viewed).astype(np.float32)
        were_sampled = ~np.isnan(viewed).all(axis=1).all(axis=1)  # find nan samples
        viewed, were_sampled = viewed[were_sampled], were_sampled[were_sampled]
        viewed, were_sampled = shuffle(viewed, were_sampled)[:max_n_EEGs_per_subj]
        return viewed, were_sampled

    def _view_an_EEG_session(
        self, eeg_session_arr, init_value=0, window_size=400, n_chs=19
    ):
        try:
            init_value = init_value % window_size
            eeg_session_arr = eeg_session_arr[init_value:]
            viewed = skimage.util.view_as_windows(
                eeg_session_arr,
                window_shape=(window_size, n_chs),
                step=window_size,
            ).squeeze()

            if viewed.ndim == 2:  # to 3D
                viewed = viewed[np.newaxis, ...]

        except Exception as e:  # when window_size is too large
            viewed = np.nan * np.ones([1, window_size, n_chs])
            print(e)

        return viewed


if __name__ == "__main__":
    ################################################################################
    ## Initialization
    ################################################################################

    LOAD_DIAGS_STR = [
        "HV",
        "AD",
        "DLB",
        "iNPH",
    ]

    # LOAD_DIAGS_STR = [
    #     "HV",
    #     "AD+DLB+iNPH",
    # ]

    # Insntanciates dataloader packer
    dlpacker = DataloaderFiller(
        i_fold=0,
        load_diags_str=LOAD_DIAGS_STR,
        do_under_sampling=False,
    )

    ################################################################################
    ## Usage
    ################################################################################
    X_tra, T_tra, S_tra, Sr_tra = dlpacker.dl_tra.dataset.tensors
    X_val, T_val, S_val, Sr_val = dlpacker.dl_val.dataset.tensors
    X_tes, T_tes, S_tes, Sr_tes = dlpacker.dl_tes.dataset.tensors
    """
    X: inputs, EEG
    T: targets, Diagnosed labels
    S: Subjects ID (global)
    Sr: Subjects ID, reallocated within the dataset
    """

    """
    ## Check if Test Subject is the same as
    ## '/storage/data/EEG/EEG_DiagnosisFromRawSignals/pickled/v4/*/*/*kCV_test_subj_indi.pkl'.
    print(dlpacker.id_allocator_subj_tes.correspondence_table)
    from utils.general import load_pkl
    lpath_pkl = '/storage/data/EEG/EEG_DiagnosisFromRawSignals/pickled/v4/IA/only_startN_to_endN/HV-DLB-iNPH-AD-5CV_test_subj_indi.pkl'
    pickled_subj_ids = load_pkl(lpath_pkl)
    print(set(pickled_subj_ids['fold_0'][0]) \
          == set(dlpacker.id_allocator_subj_tes.correspondence_table.index)) # True
    """

    """
    ## Check global subject IDs
    # len(np.unique(S_tra)) # 140
    # len(np.unique(S_val)) #  36
    # len(np.unique(S_tes)) #  59

    # len(np.unique(np.concatenate([np.unique(S_tra),
    #                       np.unique(S_val),
    #                       np.unique(S_tes)]))) # 235
    """

    ## Example 2 (recommended)
    # use as dataloader
    for epoch in range(10):
        dlpacker.fill()  # windowing from dlpacker.data_all
        dl_tra = dlpacker.dl_tra
        dl_val = dlpacker.dl_val
        dl_tes = dlpacker.dl_tes

        ## Training
        for i_batch, batch in enumerate(dl_tra):
            Xb, Tb, Sb, Srb = batch
            if i_batch % 100 == 0:
                print(Xb)

        ## Validation
        for i_batch, batch in enumerate(dl_val):
            Xb, Tb, Sb, Srb = batch
            if i_batch % 100 == 0:
                print(Xb)

        ## Test
        for i_batch, batch in enumerate(dl_tes):
            Xb, Tb, Sb, Srb = batch
            if i_batch % 100 == 0:
                print(Xb)
    """
    Xb: batched inputs, EEG
    Tb: batched targets, Diagnosed labels
    Sb: batched Subjects ID (global)
    Srb: batched Subjects ID, reallocated within the dataset
    """

    ## Example 3 (in Pytroch Lightning)
    """
    def train_dataloader(self):
        self.dlpacker.fill()
        return self.dlpacker.dl_tra

    def val_dataloader(self):
        return self.dlpacker.dl_val

    def test_dataloader(self):
        return self.dlpacker.dl_tes
    """

    ## EOF
