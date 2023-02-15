import random
from typing import Any, Dict, List, Optional, Tuple, Union

import modin.pandas as pd
import numpy as np
import ray
import torch
from ray.cloudpickle import dump, load
from torch.utils.data import DataLoader, Dataset

from quantile_discretizer import encode

ray.init()


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
        self._min_trajectory_length = 20
        self._max_trajectory_length = 120
        self._sample_dict = {}

    @property
    def sequence_length(self):
        if self._sequence_length is None:
            return random.randint(
                self._min_trajectory_length, self._max_trajectory_length
            )
        return self._sequence_length

    def __len__(self):
        return len(self._determinable_idx)

    def __getitem__(self, idx):
        dict_key = f"{idx}_{self.sequence_length}"

        if self._sample_dict.get(dict_key) is None:
            query_date = self._determinable_idx[idx]
            loc = self.df.get_loc(query_date)

            if loc <= self._max_trajectory_length + 1:
                loc = random.randint(
                    self._max_trajectory_length + 1, self.df.shape[0] - 1
                )

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
# with open("./src/assets/sequential_data.pkl", "wb") as f:
#     dump(sequential_data, f)

# load the object from the file and call a meth od on it
with open("./src/assets/sequential_data.pkl", "rb") as f:
    sequential_data = load(f)

train_data = DateReader(
    df=sequential_data.train_data,
    sequence_length=None,
    custom_index=sequential_data.train_idx,
)
dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
for i, x in enumerate(dataloader):
    print(f"Iteration {i}: x = {x}")
