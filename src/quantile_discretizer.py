from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import KBinsDiscretizer

from util import print_c


class QuantileDiscretizer:
    def __init__(self, df: pd.DataFrame, known_real: List[str], alpha: float = 3.5):
        # binary_attr = [f_type for f_type in known_real if "feature_binary" in f_type]

        df = df[known_real]

        mean = df.mean(axis=0)
        std = df.std(axis=0, ddof=1)

        # 이상치 처리
        self.lower = mean - 3 * std
        self.upper = mean + 3 * std
        # for attr in binary_attr:
        #     self.lower[attr] = -1
        #     self.upper[attr] = 1

        self.clipped_vectors = df.clip(self.lower, self.upper, axis=1)
        self.mean = self.clipped_vectors.mean()
        self.std = self.clipped_vectors.std()
        self.max = self.clipped_vectors.max()
        self.min = self.clipped_vectors.min()

        # 이산화 빈 사이즈 결정 with scott's
        bound = self.max - self.min
        # for attr in binary_attr:
        #     bound[attr] = 2

        data_length = self.clipped_vectors.shape[0]
        h = alpha * self.std * np.power(data_length, -1 / 3)

        self.n_bins = bound / h
        self.n_bins.replace(np.inf, 0, inplace=True)
        self.n_bins = self.n_bins.astype(int)

        # for attr in binary_attr:
        #     self.n_bins[attr] = 2

    def discretizer_learn_save(self, obj_fn: str):
        # discretizer = {
        #     # 이산화 모형
        #     "model": KBinsDiscretizer(
        #         n_bins=self.n_bins.values, encode="ordinal", strategy="uniform"
        #     ).fit(self.clipped_vectors.to_numpy()),
        #     "mean": self.mean.values,
        #     "std": self.std.values,
        #     "n_bins": self.n_bins.values,
        #     "lower": self.lower.values,
        #     "upper": self.upper.values,
        #     # 데이터의 순서쌍이 일치하는지 검증 키 - 간혹 버젼 os 특성으로 컬럼의 순서가 바뀌는 경우가 있음
        #     "valide_key": list(self.clipped_vectors.columns),
        # }

        discretizer = {
            # 이산화 모형
            "model": KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="uniform"
            ).fit(self.clipped_vectors.to_numpy()),
            "mean": self.mean,
            "std": self.std,
            "n_bins": self.n_bins,
            "lower": self.lower,
            "upper": self.upper,
            "vectors": self.clipped_vectors,  # Delete later
            # 데이터의 순서쌍이 일치하는지 검증 키 - 간혹 버젼 os 특성으로 컬럼의 순서가 바뀌는 경우가 있음
            "valide_key": list(self.clipped_vectors.columns),
        }
        dump(discretizer, obj_fn)

    # def quantized_vectors(self, vectors):
    #     return self.discretizer.transform(vectors)


# # Convert the quantized vectors into decimal values
# @ray.remote
# def decimal_conversion(vector, n_bins):
#     decimal = 0
#     for i, value in enumerate(vector):
#         decimal += value * (n_bins**i)
#     return decimal


# def quantized_vectors(vectors):
#     return self.discretizer.transform(vectors)
