from itertools import chain

import numpy as np
import pandas as pd

from .core import (exclude_close_events, exclude_movement, filter_ripple_band,
                   gaussian_smooth, get_envelope,
                   get_multiunit_population_firing_rate,
                   merge_overlapping_ranges, threshold_by_zscore)


def Kay_ripple_detector(time, LFPs, speed, sampling_frequency,
                        speed_threshold=4.0, minimum_duration=0.015,
                        zscore_threshold=2.0, smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Kay et al. 2016 [1].

    Parameters
    ----------
    time : array_like, shape (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) & pd.notnull(speed)
    LFPs, speed, time = LFPs[not_null], speed[not_null], time[not_null]

    filtered_lfps = np.stack(
        [filter_ripple_band(lfp, sampling_frequency) for lfp in LFPs.T])
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency)
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold)
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time,
        speed_threshold=speed_threshold)
    ripple_times = exclude_close_events(
        ripple_times, close_ripple_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)


def Karlsson_ripple_detector(time, LFPs, speed, sampling_frequency,
                             speed_threshold=4.0, minimum_duration=0.015,
                             zscore_threshold=3.0, smoothing_sigma=0.004,
                             close_ripple_threshold=0.0):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Karlsson et al. 2009 [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913-918.


    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) & pd.notnull(speed)
    LFPs, speed, time = LFPs[not_null], speed[not_null], time[not_null]

    candidate_ripple_times = []
    for lfp in LFPs.T:
        is_nan = np.isnan(lfp)
        filtered_lfp = filter_ripple_band(
            lfp, sampling_frequency=sampling_frequency)
        filtered_lfp = gaussian_smooth(
            get_envelope(filtered_lfp[~is_nan]), sigma=smoothing_sigma,
            sampling_frequency=sampling_frequency)
        lfp_ripple_times = threshold_by_zscore(
            filtered_lfp, time[~is_nan], minimum_duration,
            zscore_threshold)
        candidate_ripple_times.append(lfp_ripple_times)

    candidate_ripple_times = list(merge_overlapping_ranges(
        chain.from_iterable(candidate_ripple_times)))
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time,
        speed_threshold=speed_threshold)
    ripple_times = exclude_close_events(
        ripple_times, close_ripple_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)


def Roumis_ripple_detector(time, LFPs, speed, sampling_frequency,
                           speed_threshold=4.0, minimum_duration=0.015,
                           zscore_threshold=2.0, smoothing_sigma=0.004,
                           close_ripple_threshold=0.0):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    [1] https://bitbucket.org/franklab/trodes2ff_shared/src/b156c8d5fef3a2f89e15a678046c52919638162e/extractEventConsensus.m?at=develop&fileviewer=file-view-default

    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) & pd.notnull(speed)
    LFPs, speed, time = LFPs[not_null], speed[not_null], time[not_null]
    filtered_lfps = [filter_ripple_band(lfp, sampling_frequency)
                     for lfp in LFPs.T]
    filtered_lfps = [np.sqrt(gaussian_smooth(
        filtered_lfp ** 2, smoothing_sigma, sampling_frequency))
        for filtered_lfp in filtered_lfps]
    combined_filtered_lfps = np.mean(filtered_lfps, axis=0)
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold)
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time,
        speed_threshold=speed_threshold)
    ripple_times = exclude_close_events(
        ripple_times, close_ripple_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)


def multiunit_HSE_detector(time, multiunit, speed, sampling_frequency,
                           speed_threshold=4.0, minimum_duration=0.015,
                           zscore_threshold=3.0, smoothing_sigma=0.015,
                           close_event_threshold=0.0):
    '''Multiunit High Synchrony Event detector. Finds times when the multiunit
    population spiking activity is high relative to the average.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    speed : ndarray, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float
        Maximum running speed of animal to be counted as an event
    minimum_duration : float
        Minimum time the z-score has to stay above threshold to be
        considered an event.
    zscore_threshold : float
        Number of standard deviations the multiunit population firing rate must
        exceed to be considered an event
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.
    close_event_threshold : float
        Exclude events that occur within `close_event_threshold` of a
        previously detected event.

    Returns
    -------
    high_synchrony_event_times : pandas.DataFrame, shape (n_events, 2)

    '''
    firing_rate = get_multiunit_population_firing_rate(
        multiunit, sampling_frequency, smoothing_sigma)
    candidate_high_synchrony_events = threshold_by_zscore(
        firing_rate, time, minimum_duration, zscore_threshold)
    high_synchrony_events = exclude_movement(
        candidate_high_synchrony_events, speed, time,
        speed_threshold=speed_threshold)
    high_synchrony_events = exclude_close_events(
        high_synchrony_events, close_event_threshold)
    index = pd.Index(np.arange(len(high_synchrony_events)) + 1,
                     name='event_number')
    return pd.DataFrame(high_synchrony_events,
                        columns=['start_time', 'end_time'],
                        index=index)
