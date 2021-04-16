from glob import glob
import numpy as np
import myfunc as mf
from tqdm import tqdm


## Get file paths for olfactory bulb data
# (mouse_num, tt_num1, tt_num2, tt_num3)
olb_list = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]

def get_npy_list(olb_list):
  npy_list_tmp = []
  for day in range(1, 5):
    for i in range(len(olb_list)):
        olb_set = olb_list[i]
        mouse_num = olb_set[0]
        tt1= olb_set[1]
        # tt2= olb_set[2]
        # tt3= olb_set[3]
        tt1_npy = glob('../data/0{}/day{}/split/tt{}*.npy'\
                       .format(mouse_num, day, tt1))
        # tt2_npy = glob('../data/0{}/day{}/split/tt{}*.npy'\
        #                .format(mouse_num, day, tt2))
        # tt3_npy = glob('../data/0{}/day{}/split/tt{}*.npy'\
        #                .format(mouse_num, day, tt3))
        npy_list_tmp.append(tt1_npy)
        # npy_list_tmp.append(tt2_npy)
        # npy_list_tmp.append(tt3_npy)

  npy_list = []
  for i in range(len(npy_list_tmp)):
      max4 = npy_list_tmp[i]
      for j in range(len(max4)):
          npy_list.append(max4[j])
          # print max4[j]
  npy_list.sort()
  return npy_list

npy_list = get_npy_list(olb_list)

## Save
savepath_npy_list = '../data/npy_list.pkl'
mf.pkl_save(npy_list, savepath_npy_list)
print('Saved to {}'.format(savepath_npy_list))

## Load olbpocampal LFPs
loadpath_npy_list = savepath_npy_list # '../data/npy_list.pkl'
fpaths = mf.pkl_load(loadpath_npy_list)

def get_olb_data(fpaths):
  olb_data = []
  for i in tqdm(range(len(fpaths))):
      data_tmp = np.load(fpaths[i])
      olb_data.append(data_tmp[0,:])
  return olb_data

olb_data = get_olb_data(fpaths)

# for i in range(len(olb_data)):
#     print olb_data[i].shape
