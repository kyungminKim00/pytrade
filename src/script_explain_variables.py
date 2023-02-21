import random
from multiprocessing import Manager
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
from joblib import dump, load
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import DataReader
from preprocess import SequentialDataSet
from quantile_discretizer import QuantileDiscretizer
from util import print_c, print_flush

candle_size = [(1, 3, 5, 15, 60)]
w_size = (9, 50, 100)
# w_size = (9, 50)
alpha = [3.5]

for _candle_size in candle_size:
    # 전처리 완료 데이터
    offset = 35000  # small data or operating data
    offset = None  # practical data

    sequential_data = SequentialDataSet(
        raw_filename_min="./src/local_data/raw/dax_tm3.csv",
        pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
        candle_size=_candle_size,
        w_size=w_size,
        debug=False,
        offset=offset,
    )
    dump(sequential_data, "./src/assets/sequential_data.pkl")

    # 전처리 완료 데이터 로드
    processed_data = load("./src/assets/sequential_data.pkl")

    # 변수 설정
    x_real = [c for c in processed_data.train_data.columns if "feature" in c]

    # print_c("먼저 보기 continue 값만 먼저 보기")
    # print_c("먼저 보기 continue 값만 먼저 보기")
    # print_c("먼저 보기 continue 값만 먼저 보기")
    # x_real = [c for c in processed_data.train_data.columns if "feature_cont" in c]
    y_real = ["y_rtn_close"]

    # 이산화 모듈 저장
    for _alpha in alpha:
        # # 저장 했으면 disable
        # qd = QuantileDiscretizer(processed_data.train_data, x_real, alpha=_alpha)
        # qd.discretizer_learn_save("./src/assets/discretizer.pkl")

        # 이산화 모형 로드
        dct = load("./src/assets/discretizer.pkl")

        # DataReader configure
        manager = Manager()
        shared_dict = manager.dict()
        shared_dict.update(load("./src/assets/pattern_dict.pkl"))
        train_dataset = DataReader(
            df=processed_data.train_data,
            sequence_length=None,
            custom_index=processed_data.train_idx,
            discretizer=dct,
            known_real=x_real,
            unknown_real=y_real,
            pattern_dict=shared_dict,
        )

        batch_size = 100
        dataset = train_dataset
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        progress_bar = tqdm(dataloader)
        total_sample_about = processed_data.train_data.shape[0]
        r_num_new_pattern = 0
        for i, (x, x_date, num_new_pattern) in enumerate(progress_bar):
            r_num_new_pattern = num_new_pattern[-1].cpu()
            progress_bar.set_postfix(
                pattern=r_num_new_pattern,
                explain_pattern=(1 - (r_num_new_pattern / total_sample_about)),
            )

        # 패턴 커버리지 분석
        print_c(f"bins: {dct.n_bins} ")
        print_c(f"mean: {dct.mean} ")
        print_c(f"std: {dct.std} ")
        print_c(f"max: {dct.max} ")
        print_c(f"min: {dct.min} ")
        print_c(f"candle_size: {_candle_size}")
        print_c(f"w_size: {w_size}")
        print_c(f"alpha: {_alpha}")
        print_c(
            f"새롭게 찾은 패턴의수({r_num_new_pattern:,}) \
                샘플수 determineable_samples ({len(train_dataset):,}) \
                전체샘플수({total_sample_about:,})"
        )
        print_c(f"Score:{(1 - (r_num_new_pattern / total_sample_about))}")

assert False, "Done"


# 새롭게 추가된 패턴 저장
dump(dataset.pattern_dict, "./src/assets/pattern_dict.pkl")
print(f"dataset.pattern_dict: {dataset.pattern_dict}")


"""[검증 데이터 활용 섹션]
"""
