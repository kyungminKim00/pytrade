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

# if ray.is_initialized():
#     ray.init()


# 전처리 완료 데이터
offset = 35000  # small data or operating data
offset = None  # practical data

start = time()
sequential_data = SequentialDataSet(
    raw_filename_min="./src/local_data/raw/dax_tm3.csv",
    pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
    debug=False,
    offset=offset,
)
end = time()
print(f"\n== Time cost for [SequentialDataSet] : {end - start}")
dump(sequential_data, "./src/assets/sequential_data.pkl")

# 전처리 완료 데이터 로드
processed_data = load("./src/assets/sequential_data.pkl")

# 변수 설정
x_real = [c for c in processed_data.train_data.columns if "spd" in c]
y_real = ["y_rtn_close"]

## [지우기] 전처리 완료 데이터 저장 - 사용 하지 않음 (modin.pandas 오류시 고려하기)
# processed_data = {
#     "train_data": sequential_data.train_data._to_pandas(),
#     "validation_data": sequential_data.validation_data._to_pandas(),
#     "inference_data": sequential_data.inference_data._to_pandas(),
#     "train_idx": sequential_data.train_idx,
#     "validation_idx": sequential_data.validation_idx,
#     "inference_idx": sequential_data.inference_idx,
# }
# dump(
#     processed_data,
#     "./src/assets/sequential_data.pkl",
# )
##  [지우기] 전처리 데이터 로드 - 사용 하지 않음 (modin.pandas 오류시 고려하기)
# processed_data = load("./src/assets/sequential_data.pkl")


"""[학습 데이터 활용 섹션]
"""
# 이산화 모듈 저장
qd = QuantileDiscretizer(processed_data.train_data, x_real)
qd.discretizer_learn_save("./src/assets/discretizer.pkl")

# 이산화 모형 로드
dct = load("./src/assets/discretizer.pkl")

train_dataset = DataReader(
    df=processed_data.train_data,
    sequence_length=None,
    custom_index=processed_data.train_idx,
    discretizer=dct,
    known_real=x_real,
    unknown_real=y_real,
    pattern_dict=load("./src/assets/pattern_dict.pkl"),
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

assert False, "Done"


# 새롭게 추가된 패턴 저장
dump(dataset.pattern_dict, "./src/assets/pattern_dict.pkl")
print(f"dataset.pattern_dict: {dataset.pattern_dict}")


"""[검증 데이터 활용 섹션]
"""
