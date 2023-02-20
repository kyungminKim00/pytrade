# -*- coding: utf-8 -*-
"""
@author: kim KyungMin
"""

from __future__ import absolute_import, division, print_function

import collections
import datetime
import glob
import json
import os
import pickle
import shutil
import sys
import time
from contextlib import contextmanager
from itertools import groupby

# from memory_profiler import profile
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

import cloudpickle

# from sklearn.externals import joblib
import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from PIL import Image


def get_min_range(interval: int) -> Dict[int, int]:
    min_range = {}
    mins = np.arange(0, 61, 1)[::interval]
    for idx, val in enumerate(mins):
        if val == 0:
            pass
        else:
            min_range[idx] = [mins[idx - 1], mins[idx]]
    return min_range


def log(
    filename: str = None, log_dict: Dict[str, Any] = None, _bool: bool = True
) -> None:
    if _bool:
        fp = open(filename, "a")
        _str = " "
        for k, v in log_dict.items():
            _str = _str + f"{k}: {v}\t"
        fp.write(_str)
        fp.close()
        print(_str)


def print_warning(_str, _bool=True):
    # https://sosomemo.tistory.com/59
    if _bool:
        print("\033[31m" + _str + "\033[0m")


def print_c(_str, _bool=True):
    if _bool:
        print("\033[33m" + _str + "\033[0m")


def print_flush(_str, _bool=True):
    if _bool:
        sys.stdout.write("\r>> " + _str)
        sys.stdout.flush()


def inline_print(item, cnt, interval):
    if cnt > 0:
        if (cnt % interval) == 0:
            print(item, sep=" ", end="", flush=True)


def remove_duplicaated_dict_in_list(m_list):
    return list(
        map(
            dict,
            collections.OrderedDict.fromkeys(
                tuple(sorted(it.items())) for it in m_list
            ),
        )
    )


def get_unique_list(var_list):
    return list(map(itemgetter(0), groupby(var_list)))


def consecutive_junk(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def consecutive_true(data):
    cnt, max_cnt = 0, 0
    for i in range(len(data)):
        cnt = cnt + 1 if not data[i] else 0
        max_cnt = np.max([cnt, max_cnt])
    return max_cnt


def animate_gif(source_dir, duration=100):
    width = 640 * 2
    height = 480 * 2

    img, *imgs = None, None
    t_dir = source_dir

    # filepaths
    fp_in = t_dir + "/*.png"
    fp_out = t_dir + "/animated.gif"

    try:
        print("Reading:{}".format(fp_in))
        img, *imgs = [
            Image.open(f).resize((int(width), int(height)))
            for f in sorted(glob.glob(fp_in))
        ]
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=duration,
        )
        img.close()
    except ValueError:
        print("ValueError:{}".format(fp_in))


# if an object contains one more char type then return False
def charIsNum(string):
    bln = True
    for ch in string:
        if ord(ch) > 48 | ord(ch) < 57:
            bln = False
    return bln


def is_empty(a):
    return not a and isinstance(a, collections.Iterable)


def nanTozero(result):
    return np.where(np.isnan(result), 0, result)


# time check
@contextmanager
def funTime(func_name):
    start = time.time()
    yield
    end = time.time()
    interval = end - start
    print("\n== Time cost for [{0}] : {1}".format(func_name, interval))


def dict2json(file_name, _dict):
    # Save to json
    with open(file_name, "w") as f_out:
        json.dump(_dict, f_out)
    f_out.close()


def removeAllFile(file_directory):
    if os.path.exists(file_directory):
        for file in os.scandir(file_directory):
            try:
                os.remove(file.path)
            except IsADirectoryError:
                shutil.rmtree(file.path)
    else:
        raise FileNotFoundError


def reset_dir(file_directory):
    if not os.path.isdir(file_directory):
        os.makedirs(file_directory)
    else:
        removeAllFile(file_directory)


def json2dict(file_name):
    with open(file_name, "r") as f_out:
        _dict = json.load(f_out)
    f_out.close()
    return _dict


# Get file names in directory
def getFileList(file_directory):
    file_names = []
    for file_list in os.listdir(file_directory):
        file_names.append("{0}/{1}".format(file_directory, file_list))

    return file_names


def read_pickle(file_name):
    fp = open(file_name, "rb")
    data = pickle.load(fp)
    fp.close()
    return data


def write_pickle(data: Any, file_name: str):
    fp = open(file_name, "wb")
    data = pickle.dump(data, fp)
    fp.close()


def find_date(source_dates, str_date, search_interval):
    max_test = 1
    datetime_obj = datetime.datetime.strptime(str_date, "%Y-%m-%d")
    while True:
        return_date = np.argwhere(source_dates == datetime_obj.strftime("%Y-%m-%d"))
        if len(return_date) > 0:
            return return_date[0][0]
        else:
            datetime_obj += datetime.timedelta(days=search_interval)
            max_test = max_test + 1
        assert (
            max_test <= 10
        ), "check your date object, might be something wrong (Required Format: yyyy-mm-dd )!!!"


# Serialization with DataFrame
def pdToFile(file_name, df):
    file_name = "{0}.csv".format(file_name)
    df.to_csv(file_name, index=False)
    print("\nExporting files {0}.... Done!!!".format(file_name))


# Deserialization with DataFrame
def fileToPd(file_name):
    # Load file
    file_name = "{0}.csv".format(file_name)
    data = pd.read_csv(file_name, index_col=False)
    print("\nLoading files {0}.... Done!!!".format(file_name))
    return data


def npToFile(file_name, X, format="%s"):
    file_name = "{0}.csv".format(file_name)
    np.savetxt(file_name, X, fmt=format, delimiter=",")


def str_join(sep, *args):
    return "{}".format(sep).join(args)
