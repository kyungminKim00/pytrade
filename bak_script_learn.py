import random
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
import torch
from joblib import dump, load
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from preprocess import SequentialDataSet
from quantile_discretizer import QuantileDiscretizer
from util import print_c, print_flush

ray.shutdown()
ray.init()


# def decimal_conversion(vector, n_bins):
#     decimal = 0
#     for i, value in enumerate(vector):
#         decimal += value * (n_bins**i)
#     return decimal


class DataReader(Dataset):
    def __init__(
        self,
        df,
        custom_index,
        sequence_length=None,
        discretizer=None,
        known_real=None,
        unknown_real=None,
        pattern_dict=None,
    ):
        self._df = df[known_real]
        self._df_y = df[unknown_real]

        self._determinable_idx = custom_index
        self._sequence_length = sequence_length
        self._min_sequence_length = 60
        self._max_sequence_length = 120
        self._sample_dict = {}
        self.num_new_pattern = 0
        self.pattern_dict = pattern_dict

        self.discretizer = discretizer["model"]
        self._lower = discretizer["lower"]
        self._upper = discretizer["upper"]
        # self._n_bins = discretizer["n_bins"]
        # self._mean = discretizer["mean"]
        # self._std = discretizer["std"]
        assert (
            list(known_real) == discretizer["valide_key"]
        ), "입력의 차원의 변경이 발생 함 혹은 컬럼의 순서 변경 가능성 있음"

    @property
    def sequence_length(self):
        if self._sequence_length is None:
            return random.randint(self._min_sequence_length, self._max_sequence_length)
        return self._sequence_length

    def __len__(self):
        return len(self._determinable_idx)

    def __getitem__(self, idx):
        rcd_sequence_length = self.sequence_length
        dict_key = f"{idx}_{rcd_sequence_length}"

        if self._sample_dict.get(dict_key) is None:
            query_date = self._determinable_idx[idx]
            loc = self._df.index.get_loc(query_date)

            if loc <= self._max_sequence_length + 1:
                loc = random.randint(
                    self._max_sequence_length + 1, self._df.shape[0] - 1
                )

            self._sample_dict[dict_key] = self.encode(
                self._df.iloc[loc - self._max_sequence_length + 1 : loc + 1],
                rcd_sequence_length,
            )

        X, X_datetime = self._sample_dict[dict_key]

        # convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float)

        return X_tensor, X_datetime

    def __repr__(self):
        pass

    # convert the quantized vectors into a single integers
    def encode(self, df: pd.DataFrame, sequence_length: int) -> pd.Series:
        clipped_vectors = df.clip(self._lower, self._upper, axis=1)
        clipped_vectors = clipped_vectors[-sequence_length:]

        pattern_tuple = tuple(
            map(tuple, self.discretizer.transform(clipped_vectors).astype(int))
        )

        def id_from_dict(pttn):
            max_number = len(self.pattern_dict)

            if self.pattern_dict.get(pttn) is None:
                self.num_new_pattern = self.num_new_pattern + 1
                p_id = max_number + 1
                self.pattern_dict[pttn] = p_id
            p_id = self.pattern_dict[pttn]

            return p_id

        datetime = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        pattern_id = list(map(id_from_dict, pattern_tuple))

        # padding
        pattern_id = tuple(
            [0] * (self._max_sequence_length - sequence_length) + pattern_id
        )

        return (pattern_id, datetime)


# # 전처리 완료 데이터
# offset = 35000  # small data or operating data
# offset = None  # practical data

# start = time()
# sequential_data = SequentialDataSet(
#     raw_filename_min="./src/local_data/raw/dax_tm3.csv",
#     pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
#     debug=False,
#     offset=offset,
# )
# end = time()
# print(f"\n== Time cost for [SequentialDataSet] : {end - start}")

# dump(sequential_data, "./src/assets/sequential_data.pkl")

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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

progress_bar = tqdm(dataloader)
determineable_samples_about = len(dataloader) * batch_size
total_sample_about = determineable_samples_about * 3
for i, (x, x_date) in enumerate(progress_bar):
    progress_bar.set_postfix(
        pattern=dataset.num_new_pattern,
        explain_pattern=(1 - (dataset.num_new_pattern / total_sample_about)),
    )
# 패턴 커버리지 분석
print_c(
    f"새롭게 찾은 패턴의수({dataset.num_new_pattern}) 샘플수(determineable_samples{determineable_samples_about}) 전체샘플수({total_sample_about})"
)
print_c(f"패턴의 설명력({1 - (dataset.num_new_pattern/total_sample_about)}, max=1)")
end = time()
print(f"\n== Time cost for [Data Loader] : {end - start}")

assert False, "Done"


# 새롭게 추가된 패턴 저장
dump(dataset.pattern_dict, "./src/assets/pattern_dict.pkl")
print(f"dataset.pattern_dict: {dataset.pattern_dict}")


"""[검증 데이터 활용 섹션]
"""
