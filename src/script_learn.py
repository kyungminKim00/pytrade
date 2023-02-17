import random
from typing import Any, Dict, List, Optional, Tuple, Union

import modin.pandas as pd
import numpy as np
import ray
import torch
from joblib import dump, load
from torch.utils.data import DataLoader, Dataset

from preprocess import SequentialDataSet
from quantile_discretizer import QuantileDiscretizer

ray.shutdown()
ray.init()


def decimal_conversion(vector, n_bins):
    decimal = 0
    for i, value in enumerate(vector):
        decimal += value * (n_bins**i)
    return decimal


class DateReader(Dataset):
    def __init__(
        self,
        df,
        custom_index,
        sequence_length=None,
        discretizer=None,
        known_real=None,
        unknown_real=None,
    ):
        self._df = df[known_real]
        self._df_y = df[unknown_real]

        self._determinable_idx = custom_index
        self._sequence_length = sequence_length
        self._min_sequence_length = 20
        self._max_sequence_length = 120
        self._sample_dict = {}

        self.discretizer = discretizer["model"]
        self._lower = discretizer["lower"]
        self._upper = discretizer["upper"]
        self._n_bins = discretizer["n_bins"]
        self._mean = discretizer["mean"]
        self._std = discretizer["std"]
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
                self._df.iloc[loc - rcd_sequence_length : loc + 1]
            )

        X = self._sample_dict[dict_key]

        # convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        return X_tensor

    # convert the quantized vectors into decimal values
    def encode(self, df: pd.DataFrame) -> pd.Series:
        obj_dict = {}

        clipped_vectors = df.clip(self._lower, self._upper, axis=1)

        # 각 속성(continuous values)을 이산화하고 이산화된 벡터를 유니크한 하나의 정수로 표현한다
        # 즉, continuous_values로 표현된 패턴을 심볼릭 패턴으로 변환한다.
        decimal_vectors = [
            str(decimal_conversion(vector, self._n_bins))
            for vector in self.discretizer.transform(clipped_vectors)
        ]

        def id_from_dict(pttn):
            max_number = len(obj_dict)
            key = "_".join(pttn)

            if obj_dict.get(key) is None:
                p_id = max_number + 1
                obj_dict[key] = p_id
                dump(obj_dict, "./assets/obj_dict.pkl")
            p_id = obj_dict[key]

            return p_id

        pattern_id = list(map(id_from_dict, decimal_vectors))

        return pd.Series(pattern_id, index=df.index)


# # 전처리 완료 데이터
# sequential_data = SequentialDataSet(
#     raw_filename_min="./src/local_data/raw/dax_tm3.csv",
#     pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
#     debug=False,
# )
# dump(sequential_data, "./src/assets/sequential_data.pkl")
processed_data = load("./src/assets/sequential_data.pkl")


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


# 이산화 모형 학습과 저장
x_real = [c for c in processed_data.train_data.columns if "spd" in c]
y_real = ["y_rtn_close"]

qd = QuantileDiscretizer(processed_data.train_data, x_real)
qd.discretizer_learn_save("./src/assets/discretizer.pkl")


# 이산화 모형 로드
dct = load("./src/assets/discretizer.pkl")

train_dataset = DateReader(
    df=processed_data.train_data,
    sequence_length=None,
    custom_index=processed_data.train_idx,
    discretizer=dct,
    known_real=x_real,
    unknown_real=y_real,
)
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
for i, x in enumerate(dataloader):
    print(f"Iteration {i}: x = {x}")
