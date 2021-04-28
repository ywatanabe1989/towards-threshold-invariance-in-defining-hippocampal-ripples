#!/usr/bin/env python

import numpy as np
import re
import time
import yaml

def split_fpath(fpath):
    '''Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
    Example:
        dirname, fname, ext = split_fpath('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
        print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
        print(fname) # 'tt8-2'
        print(ext) # '.mat'
    '''
    import os
    dirname = os.path.dirname(fpath) + '/'
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext


def to_str_dtype(dtype):
    if dtype == np.int16:
        return 'int16'
    elif dtype == np.float16:
        return 'fp16'
    else:
        return None


def to_int_samp_rate(samp_rate_int):
    TO_INT_SAMP_RATE_DICT = {'2kHz':2000, '1kHz':1000, '500kHz': 500}
    return TO_INT_SAMP_RATE_DICT[samp_rate_int]


def to_str_samp_rate(samp_rate_str):
    TO_STR_SAMP_RATE_DICT = {2000: '2kHz', 1000: '1kHz', 500: '500kHz'}
    return TO_STR_SAMP_RATE_DICT[samp_rate_str]
    

def get_samp_rate_str_from_fpath(fpath):
    samp_rate_candi_str = ['2kHz', '1kHz', '500Hz']
    for samp_rate_str in samp_rate_candi_str:
        matched = re.search(samp_rate_str, fpath)
        is_matched = not (matched is None)
        if is_matched:
            return samp_rate_str

def get_samp_rate_int_from_fpath(fpath):
    return to_int_samp_rate(get_samp_rate_str_from_fpath(fpath))
        

def calc_h(data, sampling_rate):
    return len(data) / sampling_rate / 60 / 60


def save_pkl(obj, fpath):
    import pickle
    with open(fpath, 'wb') as f: # 'w'
        pickle.dump(obj, f)
    print('Saved to: {}'.format(fpath))
    

def save_npy(np_arr, fpath):
    np.save(fpath, np_arr)
    print('Saved to: {}'.format(fpath))


def load_pkl(fpath, print=False):
    import pickle
    with open(fpath, 'rb') as f: # 'r'
        obj = pickle.load(f)
        # print(obj.keys())
        return obj

def load_npy(fpath, print=False):
    arr = np.load(fpath)
    if print:    
        print('Loaded: {}'.format(fpath))
    return arr


class TimeStamper():
    def __init__(self):
        import time; self.time = time
        self.id = -1
        self.start = time.time()
        self.prev = self.start

    def __call__(self, comment):
        now = self.time.time()
        from_start = now - self.start
        
        self.from_start_hhmmss = self.time.strftime('%H:%M:%S',
                                                    self.time.gmtime(from_start))
        from_prev = now - self.prev
        
        self.from_prev_hhmmss = self.time.strftime('%H:%M:%S',
                                                   self.time.gmtime(from_prev))
        
        self.id += 1
        self.prev = now
        
        print("Time (id:{}): tot {}, prev {} [hh:mm:ss]: {}\n".format(\
              self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment))

    def get(self):
        return self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment




def read_txt(fpath):
    f = open(fpath, "r")
    read = [l.strip('\n\r') for l in f]
    f.close()
    return read


def search_str_list(str_list, search_key):
  import re
  matched_keys = []
  indi = []
  for ii, string in enumerate(str_list):
    m = re.search(search_key, string)
    if m is not None:
      matched_keys.append(string)
      indi.append(ii)
  return indi, matched_keys
      

def load_yaml_as_dict(yaml_path = './config.yaml'):
    import yaml
    config = {}
    with open(yaml_path) as f:
        _obj = yaml.safe_load(f)
        config.update(_obj)
    return config


def fix_seed(seed=42):
    # https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    
    try:
        random.seed(seed)
    except:
        pass

    
    try:
        np.random.seed(seed)
    except:
        pass

    
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)        
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass

    
    try:
        tf.random.set_seed(seed)
    except:
        pass


# def torch_to_arr(x):
#     is_arr = isinstance(x, (np.ndarray, np.generic) )
#     if is_arr: # when x is np.array
#         return x
#     if torch.is_tensor(x): # when x is torch.tensor
#         return x.detach().numpy().cpu()
    
# def save_listed_scalars_as_csv(listed_scalars, spath_csv, column_name='_',
#                                indi_suffix=None, overwrite=False):
#     '''Puts to df and save it as csv'''
#     if overwrite == True:
#         mv_to_tmp(spath_csv, L=2)
#     indi_suffix = np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
#     df = pd.DataFrame({'{}'.format(column_name):listed_scalars}
#                       , index=indi_suffix)
#     df.to_csv(spath_csv)
#     print('Saved to: {}'.format(spath_csv))

    
# def save_listed_dfs_as_csv(listed_dfs, spath_csv, indi_suffix=None, overwrite=False):
#     '''listed_dfs:
#            [df1, df2, df3, ..., dfN]. They will be written vertically in the order.
    
#        spath_csv:
#            /hoge/fuga/foo.csv
#        indi_suffix:
#            At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
#            will be added, where i is the index of the df.On the other hand,
#            when indi_suffix=None is passed, only '{}'.format(i) will be added.
#     '''
#     if overwrite == True:
#         mv_to_tmp(spath_csv, L=2)
    
#     indi_suffix = np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
#     for i, df in enumerate(listed_dfs):
#         with open(spath_csv, mode='a') as f:
#             f_writer = csv.writer(f)
#             i_suffix = indi_suffix[i]
#             f_writer.writerow(['{}'.format(indi_suffix[i])])
#         df.to_csv(spath_csv, mode='a', index=True, header=True)
#         with open(spath_csv, mode='a') as f:
#             f_writer = csv.writer(f)
#             f_writer.writerow([''])
#     print('Saved to: {}'.format(spath_csv))
    

# def take_N_percentile(data, perc=25):
#     return sorted(data)[int(len(data)*perc/100)]
    
def mv_to_tmp(fpath, L=2):
    from shutil import move
    import os
    try:
        tgt_fname = connect_str_list_with_hyphens(fpath.split('/')[-L:])
        tgt_fpath = '/tmp/{}'.format(tgt_fname)
        move(fpath, tgt_fpath)
        print('Moved to: {}'.format(tgt_fpath))
    except:
        pass

def connect_str_list_with_hyphens(str_list):
    connected = ''
    for s in str_list:
        connected += '-' + s
    return connected[1:]

def take_closest(list_obj, num_insert):
    """
    Assumes list_obj is sorted. Returns closest value to num.
    If two numbers are equally close, return the smallest number.
    list_obj = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    num = 3.5
    mf.take_closest(list_obj, num)
    # output example (3, 3)
    """
    if math.isnan(num_insert):
      return np.nan, np.nan

    pos_num_insert = bisect_left(list_obj, num_insert)

    if pos_num_insert == 0: # When the insertion is at the first position
        closest_num = list_obj[0]
        closest_pos = pos_num_insert # 0
        return closest_num, closest_pos

    if pos_num_insert == len(list_obj): # When the insertion is at the last position
        closest_num = list_obj[-1]
        closest_pos = pos_num_insert # len(list_obj)
        return closest_num, closest_pos

    else: # When the insertion is anywhere between the first and the last positions
      pos_before = pos_num_insert - 1

      before_num = list_obj[pos_before]
      after_num = list_obj[pos_num_insert]

      delta_after = abs(after_num - num_insert)
      delta_before = abs(before_num - num_insert)

      if delta_after < delta_before:
         closest_num = after_num
         closest_pos = pos_num_insert

      else: # if delta_before <= delta_after:
         closest_num = before_num
         closest_pos = pos_before

      return closest_num, closest_pos

  
def get_random_indi(data, perc=10):
    indi = np.arange(len(data))
    N_all = len(indi)
    indi_random = np.random.permutation(indi)[:int(N_all*perc/100)]
    return indi_random


def save(obj, sfname_or_spath, makedirs=True):
    '''
    Example
      save(arr, 'data.npy')
      save(df, 'df.csv')
      save(serializable, 'serializable.pkl')    
    '''
    import pickle
    import numpy as np
    import pandas as pd
    import os
    import inspect

    spath, sfname = None, None
    
    if '/' in sfname_or_spath:
        spath = sfname_or_spath
    else:
        sfname = sfname_or_spath

    if (spath is None) and (sfname is not None):
        ## for ipython
        __file__ = inspect.stack()[1].filename
        if 'ipython' in __file__:
            __file__ = '/tmp/fake.py'

        ## spath
        fpath = __file__
        fdir, fname, _ = split_fpath(fpath)
        sdir = fdir + fname + '/'
        spath = sdir + sfname
        # spath = mk_spath(sfname, makedirs=True)

    ## Make directory
    if makedirs:
        sdir = os.path.dirname(spath)
        os.makedirs(sdir, exist_ok=True)
        
    ## Saves
    # csv
    if '.csv' in spath:
        obj.to_csv(spath)
    # numpy
    if '.npy' in spath:
        np.save(obj, spath)
    # pkl
    if '.pkl' in spath:    
        with open(spath, 'wb') as s: # 'w'
            pickle.dump(obj, s)

    # png
    if '.png' in spath: # here, obj is matplotlib.plt
        obj.savefig(spath)
        obj.close()

    print('\nSaved to: {s}\n'.format(s=spath))

    
def mk_spath(sfname, makedirs=False):
    import os
    import inspect
    
    __file__ = inspect.stack()[1].filename
    if 'ipython' in __file__: # for ipython
        __file__ = '/tmp/fake.py'

    ## spath
    fpath = __file__
    fdir, fname, _ = split_fpath(fpath)
    sdir = fdir + fname + '/'
    spath = sdir + sfname

    if makedirs:
        os.makedirs(sdir, exist_ok=True)
    return spath


def load(lpath):
    import pickle
    import numpy as np
    import pandas as pd
    
    # csv
    if '.csv' in lpath:
        obj = pd.read_csv(lpath)
    # numpy
    if '.npy' in lpath:
        obj = np.load(lpath)
    # pkl
    if '.pkl' in lpath:        
        with open(lpath, 'rb') as l: # 'r'
            obj = pickle.load(l)

    return obj


def configure_mpl(plt):
    plt.rcParams['font.size'] = 20
    plt.rcParams["figure.figsize"] = (16*1.2, 9*1.2)

def makedirs_from_spath(spath):
    import os
    sdir = os.path.dirname(spath)
    os.makedirs(sdir, exist_ok=True)
