import numpy as np
from tqdm import tqdm
import utils.myfunc as mf

class pick_hip_data():
  def __init__(self):
    self.loadpath_npy_list = '../data/1000Hz_npy_list.pkl'
    self.fpaths = mf.pkl_load(loadpath_npy_list)
    self.hip_data = []
    self.riptimes = []
    print('Loading {}'.format(self.loadpath_npy_list))
    for i in tqdm(range(len(fpaths))):
        dirname, fname, ext = mf.split_fpath(self.fpaths[i])
        data_tmp = np.load(self.fpaths[i]).astype(np.float32)
        riptimes_tmp = mf.pkl_load(dirname + fname + '_riptimes.pkl')
        self.hip_data.append(data_tmp[0,:])
        self.riptimes.append(riptimes_tmp)
    self.hip_data, self.riptimes = hip_data, riptimes

  def pick_data_set():
    index = np.random.randint(len(self.hip_data))
    data, riptime = self.hip_data[index], self.riptimes[index]
    window_sec = 1
    sampling_rate = 1000
    window = window_sec * sampling_rate
    start = np.random.randint(len(data) - window - 1) # -1, fixme
    end = start + window
    end_sec = 1.0*end / sampling_rate
    input_data = data[start:end]
    next_ripple = riptime[end_sec < riptime['start_time']].head(1)
    next_ripple_num = np.array(next_ripple.index)
    next_ripple_onset_sec = np.array(next_ripple.start_time)
    next_ripple_offset_sec = np.array(next_ripple.end_time)
    onset_and_offset = np.array(next_ripple_onset_sec, next_ripple_offset_sec)
    return input_data, onset_and_offset
   # next_ripple_onset_sec, next_ripple_offset_sec

  def batch_pick(batchsize=10):
      batch_data = []
      for i in range(batchsize):
          t = None
          while t1 and t2:
              x, t = pick_data_set(self.hip_data, self.riptimes)
              batch_data.append((x, t1, t2))
      return np.array(batch_data)
