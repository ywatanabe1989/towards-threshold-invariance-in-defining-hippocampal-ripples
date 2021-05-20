#!/usr/bin/env python

import csv
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml


def gen_timestamp():
    from datetime import datetime

    now = datetime.now()
    now_str = now.strftime("%Y-%m%d-%H%M")
    return now_str


def split_fpath(fpath):
    """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
    Example:
        dirname, fname, ext = split_fpath('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
        print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
        print(fname) # 'tt8-2'
        print(ext) # '.mat'
    """
    import os

    dirname = os.path.dirname(fpath) + "/"
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext


def to_str_dtype(dtype):
    if dtype == np.int16:
        return "int16"
    elif dtype == np.float16:
        return "fp16"
    else:
        return None


def to_int_samp_rate(samp_rate_int):
    TO_INT_SAMP_RATE_DICT = {"2kHz": 2000, "1kHz": 1000, "500kHz": 500}
    return TO_INT_SAMP_RATE_DICT[samp_rate_int]


def to_str_samp_rate(samp_rate_str):
    TO_STR_SAMP_RATE_DICT = {2000: "2kHz", 1000: "1kHz", 500: "500kHz"}
    return TO_STR_SAMP_RATE_DICT[samp_rate_str]


def get_samp_rate_str_from_fpath(fpath):
    samp_rate_candi_str = ["2kHz", "1kHz", "500Hz"]
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

    with open(fpath, "wb") as f:  # 'w'
        pickle.dump(obj, f)
    print("Saved to: {}".format(fpath))


def save_npy(np_arr, fpath):
    np.save(fpath, np_arr)
    print("Saved to: {}".format(fpath))


def load_pkl(fpath, print=False):
    import pickle

    with open(fpath, "rb") as f:  # 'r'
        obj = pickle.load(f)
        # print(obj.keys())
        return obj


def load_npy(fpath, print=False):
    arr = np.load(fpath)
    if print:
        print("Loaded: {}".format(fpath))
    return arr


class TimeStamper:
    def __init__(self):
        import time

        self.time = time
        self.id = -1
        self.start = time.time()
        self.prev = self.start

    def __call__(self, comment):
        now = self.time.time()
        from_start = now - self.start

        self.from_start_hhmmss = self.time.strftime(
            "%H:%M:%S", self.time.gmtime(from_start)
        )
        from_prev = now - self.prev

        self.from_prev_hhmmss = self.time.strftime(
            "%H:%M:%S", self.time.gmtime(from_prev)
        )

        self.id += 1
        self.prev = now

        print(
            "Time (id:{}): tot {}, prev {} [hh:mm:ss]: {}\n".format(
                self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment
            )
        )

    def get(self):
        return self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment


def read_txt(fpath):
    f = open(fpath, "r")
    read = [l.strip("\n\r") for l in f]
    f.close()
    return read


def grep(str_list, search_key):
    import re

    matched_keys = []
    indi = []
    for ii, string in enumerate(str_list):
        m = re.search(search_key, string)
        if m is not None:
            matched_keys.append(string)
            indi.append(ii)
    return indi, matched_keys


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


def load_yaml_as_dict(yaml_path="./config.yaml"):
    import yaml

    config = {}
    with open(yaml_path) as f:
        _obj = yaml.safe_load(f)
        config.update(_obj)
    return config


def fix_seeds(os=None, random=None, np=None, torch=None, tf=None, seed=42, show=True):
    # https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb
    if os is not None:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)

    if random is not None:
        random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    if tf is not None:
        tf.random.set_seed(seed)

    if print:
        print("\nRandom seeds have been fixed as {}\n".format(seed))


def torch_to_arr(x):
    is_arr = isinstance(x, (np.ndarray, np.generic))
    if is_arr:  # when x is np.array
        return x
    if torch.is_tensor(x):  # when x is torch.tensor
        return x.detach().cpu().numpy()


def save_listed_scalars_as_csv(
    listed_scalars, spath_csv, column_name="_", indi_suffix=None, overwrite=False
):
    """Puts to df and save it as csv"""
    if overwrite == True:
        mv_to_tmp(spath_csv, L=2)
    indi_suffix = np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
    df = pd.DataFrame({"{}".format(column_name): listed_scalars}, index=indi_suffix)
    df.to_csv(spath_csv)
    print("Saved to: {}".format(spath_csv))


def save_listed_dfs_as_csv(listed_dfs, spath_csv, indi_suffix=None, overwrite=False):
    """listed_dfs:
        [df1, df2, df3, ..., dfN]. They will be written vertically in the order.

    spath_csv:
        /hoge/fuga/foo.csv
    indi_suffix:
        At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
        will be added, where i is the index of the df.On the other hand,
        when indi_suffix=None is passed, only '{}'.format(i) will be added.
    """
    if overwrite == True:
        mv_to_tmp(spath_csv, L=2)

    indi_suffix = np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
    for i, df in enumerate(listed_dfs):
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            i_suffix = indi_suffix[i]
            f_writer.writerow(["{}".format(indi_suffix[i])])
        df.to_csv(spath_csv, mode="a", index=True, header=True)
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            f_writer.writerow([""])
    print("Saved to: {}".format(spath_csv))


# def take_N_percentile(data, perc=25):
#     return sorted(data)[int(len(data)*perc/100)]


def mv_to_tmp(fpath, L=2):
    import os
    from shutil import move

    try:
        tgt_fname = connect_str_list_with_hyphens(fpath.split("/")[-L:])
        ts = gen_timestamp()
        tgt_fpath = "/tmp/{}-{}".format(ts, tgt_fname)
        move(fpath, tgt_fpath)
        print("(Moved to: {})".format(tgt_fpath))
    except:
        pass


def connect_str_list_with_hyphens(str_list):
    connected = ""
    for s in str_list:
        connected += "-" + s
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

    if pos_num_insert == 0:  # When the insertion is at the first position
        closest_num = list_obj[0]
        closest_pos = pos_num_insert  # 0
        return closest_num, closest_pos

    if pos_num_insert == len(list_obj):  # When the insertion is at the last position
        closest_num = list_obj[-1]
        closest_pos = pos_num_insert  # len(list_obj)
        return closest_num, closest_pos

    else:  # When the insertion is anywhere between the first and the last positions
        pos_before = pos_num_insert - 1

        before_num = list_obj[pos_before]
        after_num = list_obj[pos_num_insert]

        delta_after = abs(after_num - num_insert)
        delta_before = abs(before_num - num_insert)

        if delta_after < delta_before:
            closest_num = after_num
            closest_pos = pos_num_insert

        else:  # if delta_before <= delta_after:
            closest_num = before_num
            closest_pos = pos_before

        return closest_num, closest_pos


def get_random_indi(data, perc=10):
    indi = np.arange(len(data))
    N_all = len(indi)
    indi_random = np.random.permutation(indi)[: int(N_all * perc / 100)]
    return indi_random


def save(obj, sfname_or_spath, makedirs=True):
    """
    Example
      save(arr, 'data.npy')
      save(df, 'df.csv')
      save(serializable, 'serializable.pkl')
    """
    import inspect
    import os
    import pickle

    import numpy as np
    import pandas as pd

    spath, sfname = None, None

    if "/" in sfname_or_spath:
        spath = sfname_or_spath
    else:
        sfname = sfname_or_spath

    if (spath is None) and (sfname is not None):
        ## for ipython
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"

        ## spath
        fpath = __file__
        fdir, fname, _ = split_fpath(fpath)
        sdir = fdir + fname + "/"
        spath = sdir + sfname
        # spath = mk_spath(sfname, makedirs=True)

    ## Make directory
    if makedirs:
        sdir = os.path.dirname(spath)
        os.makedirs(sdir, exist_ok=True)

    ## Saves
    # csv
    if spath.endswith(".csv"):
        obj.to_csv(spath)
    # numpy
    if spath.endswith(".npy"):
        np.save(spath, obj)
    # pkl
    if spath.endswith(".pkl"):
        with open(spath, "wb") as s:  # 'w'
            pickle.dump(obj, s)
    # png
    if spath.endswith(".png"):
        obj.savefig(spath)
        del obj
    # tiff
    if spath.endswith(".tiff") or spath.endswith(".tif"):
        obj.savefig(spath, dpi=300, format="tiff")  # obj is matplotlib.pyplot object
        del obj
    if spath.endswith(".yaml"):
        spath_meta_yaml = self.sdir + "meta.yaml"
        with open(spath_meta_yaml, "w") as f:
            yaml.dump(meta_dict, f)

    print("\nSaved to: {s}\n".format(s=spath))


def mk_spath(sfname, makedirs=False):
    import inspect
    import os

    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = "/tmp/fake.py"

    ## spath
    fpath = __file__
    fdir, fname, _ = split_fpath(fpath)
    sdir = fdir + fname + "/"
    spath = sdir + sfname

    if makedirs:
        os.makedirs(sdir, exist_ok=True)
    return spath


def load(lpath):
    import pickle

    import numpy as np
    import pandas as pd
    import yaml

    # csv
    if lpath.endswith(".csv"):  #    if '.csv' in lpath:
        obj = pd.read_csv(lpath)
    # numpy
    if lpath.endswith(".npy"):  #    if '.npy' in lpath:
        obj = np.load(lpath)
    # pkl
    if lpath.endswith(".pkl"):  #    if '.pkl' in lpath:
        with open(lpath, "rb") as l:  # 'r'
            obj = pickle.load(l)
    # png
    if lpath.endswith(".png"):  # '.png' in lpath:
        pass

    # yaml
    if lpath.endswith(".yaml"):
        obj = {}
        with open(lpath) as f:
            _obj = yaml.safe_load(f)
            obj.update(_obj)

    return obj


def configure_mpl(plt, figscale=10, legendfontsize="xx-small"):
    updater_dict = {
        "font.size": 20,
        "figure.figsize": (round(1.62 * figscale, 1), round(1 * figscale, 1)),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.fontsize": legendfontsize,
    }

    for k, v in updater_dict.items():
        plt.rcParams[k] = v
    print("\nMatplotilb has been configured as follows:\n{}.\n".format(updater_dict))


def makedirs_from_spath(spath):
    import os

    sdir = os.path.dirname(spath)
    os.makedirs(sdir, exist_ok=True)


def listed_dict(keys=None):
    dict_list = defaultdict(list)
    # initialize with keys if possible
    if keys is not None:
        for k in keys:
            dict_list[k] = []
    return dict_list


class Tee(object):
    """Example:
    import sys
    sys.stdout = Tee(sys.stdout, "stdout.txt")
    sys.stderr = Tee(sys.stderr, "stderr.txt")
    print("abc") # stdout
    print(1 / 0) # stderr
    # cat stdout.txt
    # cat stderr.txt
    """

    def __init__(self, sys_stdout_or_stderr, spath):
        self._files = [sys_stdout_or_stderr, open(spath, "w")]

    def __getattr__(self, attr, *args):
        return self._wrap(attr, *args)

    def _wrap(self, attr, *args):
        def g(*a, **kw):
            for f in self._files:
                res = getattr(f, attr, *args)(*a, **kw)
            return res

        return g


def tee(sys, spath=None):
    """
    import sys
    sys.stdout, sys.stderr = tee(sys)
    print("abc")  # stdout
    print(1 / 0)  # stderr
    """

    import inspect
    import os

    ####################
    ## Determines spath
    ####################
    if spath is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"
        spath = __file__
    else:
        os.makedirs(os.path.dirname(spath), exist_ok=True)

    root, ext = os.path.splitext(spath)

    ## Checks spath ext
    permitted_exts_list = [".txt", ".log"]
    if not ext in permitted_exts_list:
        root = root + ext.replace(".", "_")
        ext = ".log"

    spath_stdout = root + "_stdout" + ext
    spath_stderr = root + "_stderr" + ext
    sys_stdout = Tee(sys.stdout, spath_stdout)
    sys_stderr = Tee(sys.stdout, spath_stderr)

    print(
        "\nStandard Output/Error are going to be logged in the followings: \n  - {}\n  - {}\n".format(
            spath_stdout, spath_stderr
        )
    )
    return sys_stdout, sys_stderr
