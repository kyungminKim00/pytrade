from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import KBinsDiscretizer
from collections import OrderedDict
from util import print_c


class QuantileDiscretizer:
    def __init__(self, df: pd.DataFrame, known_real: List[str], alpha: float = 3.5):
        # binary_attr = [f_type for f_type in known_real if "feature_binary" in f_type]

        df = df[known_real]

        mean = df.mean(axis=0)
        std = df.std(axis=0, ddof=1)

        # 이상치 처리
        print_c("QuantileDiscretizer 이상치 처리 테스트 나중에 지우기")
        self.lower = mean - 8 * std
        self.upper = mean + 8 * std
        # for attr in binary_attr:
        #     self.lower[attr] = -1
        #     self.upper[attr] = 1

        self.clipped_vectors = df.clip(self.lower, self.upper, axis=1)
        self.mean = self.clipped_vectors.mean()
        self.std = self.clipped_vectors.std()
        self.max = self.clipped_vectors.max()
        self.min = self.clipped_vectors.min()

        # 추가
        self.n_bins = self.evaluate_bin_size(self.clipped_vectors, known_real, alpha)
        self.n_bins.replace(np.inf, 0, inplace=True)

    def evaluate_bin_size(self, x, known_real, alpha):
        num_of_bins = []
        if x.shape[0] > 50000:
            num_statistics = 50000
        else:
            num_statistics = x.shape[0]
        for col in known_real:
            a_data = x[col]

            partial_bin = []
            for i in range(0, len(a_data), num_statistics):
                if i == 0:
                    start = 0
                    end = num_statistics
                else:
                    start = i
                    if i + num_statistics >= len(a_data):
                        end = len(a_data)
                    else:
                        end = i + num_statistics
                partial_bin.append(self.get_bin_size(a_data[start:end], alpha))
            avg_bins = np.mean(partial_bin)
            num_of_bins.append(
                min(int(np.ceil((a_data.max() - a_data.min()) / avg_bins)), 5000)
            )
        return pd.Series(OrderedDict(zip(known_real, num_of_bins)))

    def get_bin_size(self, a_data, alpha):
        return alpha * np.std(a_data) / (len(a_data) ** (1 / 3))

    def discretizer_learn_save(self, obj_fn: str):
        # discretizer = {
        #     # 이산화 모형
        #     "model": KBinsDiscretizer(
        #         n_bins=self.n_bins.values, encode="ordinal", strategy="quantile"
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
            # "model": KBinsDiscretizer(
            #     n_bins=self.n_bins, encode="ordinal", strategy="uniform"
            # ).fit(self.clipped_vectors.to_numpy()),
            "model": KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="uniform"
            ).fit(self.clipped_vectors),
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
