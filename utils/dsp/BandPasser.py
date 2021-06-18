#!/usr/bin/env python

from scipy.signal import butter, sosfilt, sosfreqz


def BandPasser(data, lo_hz, hi_hz, fs, order=5):
    def mk_butter_bandpass(order=5):

        nyq = 0.5 * fs
        low, high = lo_hz / nyq, hi_hz / nyq
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    def butter_bandpass_filter(data):

        sos = mk_butter_bandpass()
        y = sosfilt(sos, data)
        return y

    sos = mk_butter_bandpass(order=order)
    y = butter_bandpass_filter(data)

    return y
