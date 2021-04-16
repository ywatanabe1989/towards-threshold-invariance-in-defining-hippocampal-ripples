import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')
import utils.myfunc as mf

loadpath_npy_list = '../data/2kHz_npy_list.pkl'
fpaths = mf.pkl_load(loadpath_npy_list)
# hip_data, riptimes = mf.get_hip_data(fpaths)


class pick_hip_data():
  def __init__(self, fpaths):
    hip_data = []
    riptimes = []
    for i in tqdm(range(len(fpaths))):
        dirname, fname, ext = mf.split_fpath(fpaths[i])
        lpath_data = fpaths[i]
        data_tmp = np.load(lpath_data).astype(np.float32)

        riptimes_tmp = mf.pkl_load(dirname + fname + '_riptimes.pkl')
        hip_data.append(data_tmp[0,:])
        riptimes.append(riptimes_tmp)
    self.hip_data = hip_data
    self.riptimes = rip_times

  def pick_data_set(hip_data, riptimes):
    index = np.random.randint(len(hip_data))
    data, riptime = hip_data[index], riptimes[index]
    window_sec = 1
    sampling_rate = 1000
    window = window_sec * sampling_rate
    start = np.random.randint(len(data) - window)
    end = start + window
    end_sec = 1.0*end / sampling_rate
    input_data = data[start:end]
    next_ripple = riptime[end_sec < riptime['start_time']].head(1) # [:1]
    next_ripple_num = np.array(next_ripple.index)
    next_ripple_onset_sec = np.array(next_ripple.start_time)
    next_ripple_offset_sec = np.array(next_ripple.end_time)
    return input_data, next_ripple_onset_sec, next_ripple_offset_sec

  def batch_pick(batchsize, hip_data, riptimes):
      batch_data = []
      for i in range(batchsize):
          x, t1, t2 = pick_data_set(hip_data, riptimes)
          batch_data.append((x, t1, t2))
      return np.array(batch_data)
