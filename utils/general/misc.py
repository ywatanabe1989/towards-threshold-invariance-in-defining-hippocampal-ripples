#!/usr/bin/env python

import re
import time
from collections import defaultdict

import numpy as np
import torch


################################################################################
## strings
################################################################################
def connect_strs(strs, filler="_"):
    """
    Example:
        print(connect_strs(['a', 'b', 'c'], filler='_'))
        # 'a_b_c'
    """
    if isinstance(strs, list) or isinstance(strs, tuple):
        connected = ""
        for s in strs:
            connected += filler + s
        return connected[len(filler) :]
    else:
        return strs


def squeeze_spaces(string, pattern=" +", repl=" "):
    """Return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement repl.  repl can be either a string or a callable;
    if a string, backslash escapes in it are processed.  If it is
    a callable, it's passed the Match object and must return
    a replacement string to be used.
    """
    # return re.sub(" +", " ", string)
    return re.sub(pattern, repl, string)


def grep(str_list, search_key):
    """
    Example:
        str_list = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        search_key = 'orange'
        print(grep(str_list, search_key))
        # ([1, 5], ['orange', 'orange_juice'])
    """
    matched_keys = []
    indi = []
    for ii, string in enumerate(str_list):
        m = re.search(search_key, string)
        if m is not None:
            matched_keys.append(string)
            indi.append(ii)
    return indi, matched_keys


def pop_keys(keys_list, keys_to_pop):
    """
    keys_list = ['a', 'b', 'c', 'd', 'e', 'bde']
    keys_to_pop = ['b', 'd']
    pop_keys(keys_list, keys_to_pop)
    """
    indi_to_remain = [k not in keys_to_pop for k in keys_list]
    keys_remainded_list = list(np.array(keys_list)[list(indi_to_remain)])
    return keys_remainded_list


################################################################################
## dictionary
################################################################################
def listed_dict(keys=None):  # Is there a better name?
    """
    Example 1:
        import random
        random.seed(42)
        d = listed_dict()
        for _ in range(10):
            d['a'].append(random.randint(0, 10))
        print(d)
        # defaultdict(<class 'list'>, {'a': [10, 1, 0, 4, 3, 3, 2, 1, 10, 8]})

    Example 2:
        import random
        random.seed(42)
        keys = ['a', 'b', 'c']
        d = listed_dict(keys)
        for _ in range(10):
            d['a'].append(random.randint(0, 10))
            d['b'].append(random.randint(0, 10))
            d['c'].append(random.randint(0, 10))
        print(d)
        # defaultdict(<class 'list'>, {'a': [10, 4, 2, 8, 6, 1, 8, 8, 8, 7],
        #                              'b': [1, 3, 1, 1, 0, 3, 9, 3, 6, 9],
        #                              'c': [0, 3, 10, 9, 0, 3, 0, 10, 3, 4]})
    """
    dict_list = defaultdict(list)
    # initialize with keys if possible
    if keys is not None:
        for k in keys:
            dict_list[k] = []
    return dict_list
