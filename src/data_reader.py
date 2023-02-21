import random
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

on_his_delete = True
histogram_dict = {}


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

        X, X_datetime, new_pattern = self._sample_dict[dict_key]

        # convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float)

        return X_tensor, X_datetime, new_pattern

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

            if on_his_delete:
                if histogram_dict.get(p_id) is None:
                    histogram_dict[p_id] = 0
                else:
                    histogram_dict[p_id] = histogram_dict[p_id] + 1

            return p_id

        datetime = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        pattern_id = list(map(id_from_dict, pattern_tuple))

        # padding
        pattern_id = tuple(
            [0] * (self._max_sequence_length - sequence_length) + pattern_id
        )

        return (pattern_id, datetime, self.num_new_pattern)
