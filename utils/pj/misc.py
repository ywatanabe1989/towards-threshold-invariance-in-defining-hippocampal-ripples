#!/usr/bin/env python

import re


def to_int_samp_rate(samp_rate_int):
    TO_INT_SAMP_RATE_DICT = {"2kHz": 2000, "1kHz": 1000, "500kHz": 500}
    return TO_INT_SAMP_RATE_DICT[samp_rate_int]


def to_str_samp_rate(samp_rate_str):
    TO_STR_SAMP_RATE_DICT = {2000: "2kHz", 1000: "1kHz", 500: "500kHz"}
    return TO_STR_SAMP_RATE_DICT[samp_rate_str]


def get_samp_rate_str_from_fpath(fpath):
    samp_rate_candi_str = ["2kHz", "1kHz", "500Hz"]
    for samp_rate_str in samp_rate_candi_str:
        matched = re.search(samp_rate_str, fpath)
        is_matched = not (matched is None)
        if is_matched:
            return samp_rate_str


def get_samp_rate_int_from_fpath(fpath):
    return to_int_samp_rate(get_samp_rate_str_from_fpath(fpath))


def calc_h(data, sampling_rate):
    return len(data) / sampling_rate / 60 / 60
