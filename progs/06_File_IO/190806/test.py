import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')
import utils.myfunc as mf

loadpath_npy_list = '../data/2kHz_npy_list.pkl'
fpaths = mf.pkl_load(loadpath_npy_list)
# hip_data, riptimes = mf.get_hip_data(fpaths)


class get_lfp_and_rip():
  def __init__(self, fpaths, samp_rate=1000):
    self.lfp = []
    self.rip_secs = []
    self.samp_rate = samp_rate

    for i in tqdm(range(len(fpaths))):
        lpath_lfp = fpaths[i]

        if self.samp_rate == 2000:
          pass
        if self.samp_rate == 1000:
          lpath_lfp.replace('2kHz', '1kHz')
        if self.samp_rate == 500:
          lpath_lfp.replace('2kHz', '500Hz')

        lfp_tmp, rip_sec_tmp, samp_rate = mf.get_lfp_and_sampling_rate(lpath_lfp)
        self.lfp.append(lfp_tmp)
        self.rip_secs.append(rip_sec_tmp)
    self.samp_rate = samp_rate

hip = get_lfp_and_rip(fpaths, samp_rate=1000)









  # def pick_data_set(hip_data, riptimes):
  #   index = np.random.randint(len(hip_data))
  #   data, riptime = hip_data[index], riptimes[index]
  #   window_sec = 1
  #   sampling_rate = 1000
  #   window = window_sec * sampling_rate
  #   start = np.random.randint(len(data) - window)
  #   end = start + window
  #   end_sec = 1.0*end / sampling_rate
  #   input_data = data[start:end]
  #   next_ripple = riptime[end_sec < riptime['start_time']].head(1) # [:1]
  #   next_ripple_num = np.array(next_ripple.index)
  #   next_ripple_onset_sec = np.array(next_ripple.start_time)
  #   next_ripple_offset_sec = np.array(next_ripple.end_time)
  #   return input_data, next_ripple_onset_sec, next_ripple_offset_sec

  # def batch_pick(batchsize, hip_data, riptimes):
  #     batch_data = []
  #     for i in range(batchsize):
  #         x, t1, t2 = pick_data_set(hip_data, riptimes)
  #         batch_data.append((x, t1, t2))
  #     return np.array(batch_data)
