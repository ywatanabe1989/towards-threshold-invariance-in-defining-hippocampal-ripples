#!/usr/bin/env python


def load(lpath, show=False):
    import pickle

    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    # csv
    if lpath.endswith(".csv"):
        obj = pd.read_csv(lpath)
    # numpy
    if lpath.endswith(".npy"):
        obj = np.load(lpath)
    # pkl
    if lpath.endswith(".pkl"):
        with open(lpath, "rb") as l:
            obj = pickle.load(l)
    # joblib
    if lpath.endswith(".joblib"):
        with open(lpath, "rb") as l:
            obj = joblib.load(l)
    # hdf5
    if lpath.endswith(".hdf5"):
        obj = {}
        with h5py.File(fpath, "r") as hf:
            for name in name_list:
                obj_tmp = hf[name][:]
                obj[name] = obj_tmp
    # png
    if lpath.endswith(".png"):
        pass
    # tiff
    if lpath.endswith(".tiff") or lpath.endswith(".tif"):
        pass
    # yaml
    if lpath.endswith(".yaml"):
        obj = {}
        with open(lpath) as f:
            obj_tmp = yaml.safe_load(f)
            obj.update(obj_tmp)
    # txt
    if lpath.endswith(".txt"):
        f = open(lpath, "r")
        obj = [l.strip("\n\r") for l in f]
        f.close()
    # pth
    if lpath.endswith(".pth"):
        # return model.load_state_dict(torch.load(lpath))
        obj = torch.load(lpath)

    if show:
        print("\nLoaded: {}\n".format(lpath))

    return obj


def check_encoding(file_path):
    from chardet.universaldetector import UniversalDetector

    detector = UniversalDetector()
    with open(file_path, mode="rb") as f:
        for binary in f:
            detector.feed(binary)
            if detector.done:
                break
    detector.close()
    enc = detector.result["encoding"]
    return enc
