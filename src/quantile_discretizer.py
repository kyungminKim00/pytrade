from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import KBinsDiscretizer


class QuantileDiscretizer:
    def __init__(self, df: pd.DataFrame, known_real: List[str]):

        # # 특징 변수(Attributes)의 prefix를 spd 로 지정 해 둠 (전처리 단계 - 특징생성 모듈)
        # drop_column = [c for c in df.columns if not "spd" in c]
        # df = df.drop(columns=drop_column)

        df = df[known_real]

        self.mean = df.mean(axis=0)
        self.std = df.std(axis=0, ddof=1)

        # 이상치 처리
        self.lower = self.mean - 3 * self.std
        self.upper = self.mean + 3 * self.std
        self.clipped_vectors = df.clip(self.lower, self.upper, axis=1)

        # 이산화 빈 사이즈 결정 with scott's
        data_length = df.shape[0]
        bound = df.max() - df.min()
        h = 3.5 * self.std * np.power(data_length, -1 / 3)
        self.n_bins = (bound / h).astype(int)

    def discretizer_learn_save(self, obj_fn: str):
        discretizer = {
            # 이산화 모형
            "model": KBinsDiscretizer(
                n_bins=self.n_bins.values, encode="ordinal", strategy="uniform"
            ).fit(self.clipped_vectors.to_numpy()),
            # 이상치 처리 및 패턴 인덱스 생성을 위한 파라미터
            # "mean": self.mean.values,
            # "std": self.std.values,
            # "n_bins": self.n_bins.values,
            "lower": self.lower.values,
            "upper": self.upper.values,
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
