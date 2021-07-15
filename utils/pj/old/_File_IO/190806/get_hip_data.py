from glob import glob
import numpy as np
import myfunc as mf
from tqdm import tqdm


## Get file paths for hippocampal data
# (mouse_num, tt_num1, tt_num2, tt_num3)
hip_list = [(1, 2, 6, 0), (2, 2, 6, 7), (3, 2, 6, 7), (4, 2, 6, 7), (5, 2, 6, 0)]

def get_npy_list(hip_list):
  npy_list_tmp = []
  for day in range(1, 5):
    for i in range(len(hip_list)):
        hip_set = hip_list[i]
        mouse_num = hip_set[0]
        tt1= hip_set[1]
        tt2= hip_set[2]
        tt3= hip_set[3]
        tt1_npy = glob('../data/0{}/day{}/split/tt{}*.npy'\
                       .format(mouse_num, day, tt1))
        tt2_npy = glob('../data/0{}/day{}/split/tt{}*.npy'\
                       .format(mouse_num, day, tt2))
        tt3_npy = glob('../data/0{}/day{}/split/tt{}*.npy'\
                       .format(mouse_num, day, tt3))
        npy_list_tmp.append(tt1_npy)
        npy_list_tmp.append(tt2_npy)
        npy_list_tmp.append(tt3_npy)

  npy_list = []
  for i in range(len(npy_list_tmp)):
      max4 = npy_list_tmp[i]
      for j in range(len(max4)):
          npy_list.append(max4[j])
          # print max4[j]
  npy_list.sort()
  return npy_list

npy_list = get_npy_list(hip_list)

## Save
savepath_npy_list = '../data/npy_list.pkl'
mf.pkl_save(npy_list, savepath_npy_list)
print('Saved to {}'.format(savepath_npy_list))

## Load hippocampal LFPs
loadpath_npy_list = savepath_npy_list # '../data/npy_list.pkl'
fpaths = mf.pkl_load(loadpath_npy_list)

def get_hip_data(fpaths):
  hip_data = []
  for i in tqdm(range(len(fpaths))):
      data_tmp = np.load(fpaths[i])
      hip_data.append(data_tmp[0,:])
  return hip_data

hip_data = get_hip_data(fpaths)

# for i in range(len(hip_data)):
#     print hip_data[i].shape
