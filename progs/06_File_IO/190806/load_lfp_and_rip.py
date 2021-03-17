import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')
import utils.myfunc as mf
import torch


def load_lfp_and_rip(fpaths, samp_rate=1000): # input: npy, pkl (pands.DataFrame) -> output: torch.Tensor
  # def __init__(self, fpaths, samp_rate=1000):
  lfp = []
  rip_sec = []

  for i in tqdm(range(len(fpaths))):
      lpath_lfp = fpaths[i]

      if samp_rate == 2000:
        pass
      if samp_rate == 1000:
        lpath_lfp.replace('2kHz', '1kHz')
      if samp_rate == 500:
        lpath_lfp.replace('2kHz', '500Hz')

      lfp_tmp, rip_sec_tmp, samp_rate = mf.get_lfp_rip_sec_samprate(lpath_lfp)
      print('Loaded : {}'.format(lpath_lfp))
      rip_sec_tmp = np.array(rip_sec_tmp)
      lfp.append(lfp_tmp)
      rip_sec.append(rip_sec_tmp)

  return lfp, rip_sec, samp_rate
  # self.lfp = torch.nn.utils.rnn.pad_sequence(self.lfp, batch_first=True, padding_value=-float("Inf"))
    # self.rip_sec = torch.nn.utils.rnn.pad_sequence(self.rip_sec, batch_first=True, padding_value=-float("Inf"))

# lfp, rip_sec, samp_rate = load_lfp_and_rip(samp_rate=1000)
# np.save('../data/lfp_1kHz.npy', lfp)
# np.save('../data/rip_sec.npy', rip_sec)







# class get_lfp_and_rip(): # input: npy, pkl (pands.DataFrame) -> output: torch.Tensor
#   def __init__(self, fpaths, samp_rate=1000):
#     self.lfp = []
#     self.rip_sec = []
#     self.samp_rate = samp_rate

#     for i in tqdm(range(len(fpaths))):
#         lpath_lfp = fpaths[i]

#         if self.samp_rate == 2000:
#           pass
#         if self.samp_rate == 1000:
#           lpath_lfp.replace('2kHz', '1kHz')
#         if self.samp_rate == 500:
#           lpath_lfp.replace('2kHz', '500Hz')

#         lfp_tmp, rip_sec_tmp, samp_rate = mf.get_lfp_rip_sec_samprate(lpath_lfp)
#         # n_rip, rip_eve_hz, rip_sec_tmp = mf.calc_some_metrices_from_lfp(lfp_tmp, rip_sec_tmp, samp_rate) # added
#         # lfp_tmp = torch.tensor(lfp_tmp)
#         # rip_sec_tmp = torch.tensor(np.array(rip_sec_tmp))
#         rip_sec_tmp = np.array(rip_sec_tmp)
#         self.lfp.append(lfp_tmp)
#         self.rip_sec.append(rip_sec_tmp)

#     self.samp_rate = samp_rate
#     # self.lfp = torch.nn.utils.rnn.pad_sequence(self.lfp, batch_first=True, padding_value=-float("Inf"))
#     # self.rip_sec = torch.nn.utils.rnn.pad_sequence(self.rip_sec, batch_first=True, padding_value=-float("Inf"))

# loadpath_npy_list = '../data/2kHz_npy_list.pkl'
# fpaths = mf.pkl_load(loadpath_npy_list)
# hip = get_lfp_and_rip(fpaths, samp_rate=1000)

# np.save('../data/lfp_1kHz.npy', hip.lfp)
# np.save('../data/rip_sec.npy', hip.rip_sec)
# # torch.save(hip.lfp, '../data/lfp_1kHz.pt')
# # torch.save(hip.rip_sec, '../data/rip_sec.pt')




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
