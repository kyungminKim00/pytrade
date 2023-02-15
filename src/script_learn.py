import random
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import modin.pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess import SequentialDataSet
from quantile_discretizer import encode


class DateReader(Dataset):
    def __init__(
        self,
        df,
        custom_index,
        sequence_length=None,
    ):
        self.df = df

        self._determinable_idx = custom_index
        self._sequence_length = sequence_length
        self._sample_dict = {}

    @property
    def sequence_length(self):
        if self._sequence_length is None:
            return random.randint(20, 120)
        return self._sequence_length

    def __len__(self):
        return len(self._determinable_idx)

    def __getitem__(self, idx):
        dict_key = f"{idx}_{self.sequence_length}"

        if self._sample_dict.get(dict_key) is None:
            query_date = self._determinable_idx[idx]
            loc = self.df.get_loc(query_date)

            self._sample_dict[dict_key] = encode(
                self.df.iloc[loc - self.sequence_length : loc + 1]
            )

        X = self._sample_dict[dict_key]

        # convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        return X_tensor


# # 전처리 완료 데이터
# sequential_data = SequentialDataSet(
#     raw_filename_min="./src/local_data/raw/dax_tm3.csv",
#     pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
#     debug=False,
# )
# joblib.dump(sequential_data, "./src/assets/sequential_data.pkl")

sequential_data = joblib.load("./src/assets/sequential_data.pkl")
train_data = DateReader(
    df=sequential_data.train_data,
    sequence_length=None,
    custom_index=sequential_data.train_idx,
)
print(train_data)
