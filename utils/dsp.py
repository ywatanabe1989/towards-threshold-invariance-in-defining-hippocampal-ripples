#!/usr/bin/env python

import numpy as np
from modules.rippledetection.core import gaussian_smooth

def bandpass(data, lo_hz, hi_hz, fs, order=5):

  def mk_butter_bandpass(order=5):
    from scipy.signal import butter, sosfilt, sosfreqz
    nyq = 0.5 * fs
    low, high = lo_hz/nyq, hi_hz/nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

  def butter_bandpass_filter(data):
    from scipy.signal import butter, sosfilt, sosfreqz
    sos = mk_butter_bandpass()
    y = sosfilt(sos, data)
    return y

  sos = mk_butter_bandpass(order=order)
  y = butter_bandpass_filter(data)

  return y



def calc_band_magnitude(data, samp_rate, lo_hz, hi_hz,
                        devide_by_std=False,
                        minimum_duration=0.15,
                        zscore_threshold=2.0,
                        smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):

    if (lo_hz, hi_hz) != (None, None):
        filted = bandpass(data, lo_hz, hi_hz, samp_rate)
    else:
        filted = data

    power = filted ** 2
    smoothed_power = gaussian_smooth(power, smoothing_sigma, samp_rate)
    magnitude = np.sqrt(smoothed_power)

    if devide_by_std:
        magnitude /= magnitude.std() # Normalize

    return magnitude

  
def wavelet(wave, samp_rate, f_min=100, f_max=None, plot=False):
  dt = 1. / samp_rate
  npts = len(wave)
  t = np.linspace(0, dt * npts, npts)
  if f_min == None:
      f_min = 0.1
  if f_max == None:
      f_max = int(samp_rate/2)
  scalogram = cwt(wave, dt, 8, f_min, f_max)

  if plot:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      x, y = np.meshgrid(
          t,
          np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

      ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
      ax.set_xlabel("Time [s]")
      ax.set_ylabel("Frequency [Hz]")
      ax.set_yscale('log')
      ax.set_ylim(f_min, f_max)
      plt.show()

  Hz = pd.Series(np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
  df = pd.DataFrame(np.abs(scalogram))
  df['Hz'] = Hz
  df.set_index('Hz', inplace=True)

  return df  