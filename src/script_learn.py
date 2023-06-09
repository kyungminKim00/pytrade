import random
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import ray
import torch
from joblib import dump, load
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import DataReader
from preprocess import SequentialDataSet
from quantile_discretizer import QuantileDiscretizer
from util import print_c, print_flush

import json
import os

import imageio

import plotly.express as px


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from plotly.subplots import make_subplots
from gen_feature_label import FeatureLabel
from masked_language_model import MaskedLanguageModel, MaskedLanguageModelDataset
from util import print_c, remove_files


if __name__ == "__main__":
    # 모듈 정보
    with open("./src/context_spread.json", "r", encoding="utf-8") as fp:
        env_dict = json.load(fp)

    loss_dict = {
        # "SmoothL1Loss": nn.SmoothL1Loss(),
        "mse": nn.MSELoss(),
        # "Earthmover": Earthmover,
        # "L1Loss": nn.L1Loss(),
    }
    # # 전처리 완료 데이터
    # offset = 35000  # small data or operating data
    # offset = None  # practical data

    # start = time()
    # sequential_data = SequentialDataSet(
    #     raw_filename_min=env_dict["raw_filename_min"],
    #     pivot_filename_day=env_dict["pivot_filename_day"],
    #     debug=False,
    #     offset=offset,
    # )
    # end = time()
    # print(f"\n== Time cost for [SequentialDataSet] : {end - start}")

    # 전처리 완료 데이터 저장 and 로드
    # dump(sequential_data, env_dict["sequential_data"])
    data_cls = load(env_dict["sequential_data"])

    # 변수 설정
    x_real = [c for c in data_cls.processed_data.columns if "feature" in c]
    y_real = ["y_rtn_close"]

    # # 특징 및 마스크 설정
    # print("Ray initialized already" if ray.is_initialized() else ray.init())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tv_ratio = env_dict["tv_ratio"]
    # 특징 추출
    cc = FeatureLabel(
        data_cls.processed_data,
        debug=False,
        # data_tranform=None,  # None: 데이터변환 수행안함, n_components: np.inf 는 전체 차원
        data_tranform={
            "n_components": env_dict["n_components"],
            "method": env_dict["method"],
        },  # None: 데이터변환 수행안함, n_components: np.inf 는 전체 차원
        ratio=env_dict[
            "rdt_tv_ratio"
        ],  # data_tranform is not None 일 때 PCA 학습 샘플, 차원 축소된 observation 은 전체 샘플 다 포함 함
        alpha=1.5,
        env_dict=env_dict,
        gen_expected_rtn=True,
    )

    dump(cc, f"{env_dict['feature_label']}")
    cc = load(f"{env_dict['feature_label']}")
    data = cc.observatoins_merge_idx
    weight_vars = cc.weight_variables
    padding_torken = cc.padding_torken
    forward_label = cc.forward_label

    """[학습 데이터 활용 섹션]
    """
    # 이산화 모듈 저장
    qd = QuantileDiscretizer(processed_data.train_data, x_real)
    qd.discretizer_learn_save("./src/local_data/assets/discretizer.pkl")

    # 이산화 모형 로드
    dct = load("./src/local_data/assets/discretizer.pkl")

    # load pattern dict
    try:
        pattern_dict = load("./src/local_data/assets/pattern_dict.pkl")
    except FileNotFoundError:
        pattern_dict = {}

    train_dataset = DataReader(
        df=processed_data.train_data,
        sequence_length=None,
        custom_index=processed_data.train_idx,
        discretizer=dct,
        known_real=x_real,
        unknown_real=y_real,
        pattern_dict=pattern_dict,
    )

    start = time()
    batch_size = 2
    dataset = train_dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=psutil.cpu_count(logical=False),
    )

    progress_bar = tqdm(dataloader)
    determineable_samples_about = len(dataloader) * batch_size
    total_sample_about = determineable_samples_about * 3
    r_num_new_pattern = 0
    for i, (x, x_date, num_new_pattern) in enumerate(progress_bar):
        r_num_new_pattern = num_new_pattern[-1].cpu()
        progress_bar.set_postfix(
            pattern=r_num_new_pattern,
            explain_pattern=(1 - (r_num_new_pattern / total_sample_about)),
        )

    # 패턴 커버리지 분석
    print_c(
        f"새롭게 찾은 패턴의수({r_num_new_pattern}) 샘플수 determineable_samples ({determineable_samples_about}) 전체샘플수({total_sample_about})"
    )
    print_c(f"패턴의 설명력({1 - (r_num_new_pattern/total_sample_about)}, max=1)")
    end = time()
    print(f"\n== Time cost for [Data Loader] : {end - start}")

    # 새롭게 추가된 패턴 저장
    dump(dataset.pattern_dict, "./src/local_data/assets/pattern_dict.pkl")
    print(f"dataset.pattern_dict: {dataset.pattern_dict}")

    """[검증 데이터 활용 섹션]
    """
