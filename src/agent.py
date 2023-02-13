import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import modin.pandas as pd
import numpy as np

from data_reader import DataReader
from util import print_c

actions = {
    "NP": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "BP_Init": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SP_Init": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "BP_Hold": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "SP_Hold": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "BP_Clear": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "SP_Clear": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "UK_BP_Hold": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "UK_SP_Hold": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "UK_BP_Clear": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "UK_SP_Clear": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


class Agent:
    def __init__(
        self,
        mode: str = None,
        data_reader: DataReader = None,
    ) -> None:
        # super().__init__(
        #     raw_filename_min=raw_filename_min,
        #     pivot_filename_day=pivot_filename_day,
        # )
        self._mode = mode
        self._data_reader = data_reader
        self._previous_action = actions["NP"]  # 관측 액션: T-1 시점의 액션
        self._episode_cum_return = 0
        self._account = OrderedDict()  # validation / inference 에 활용

        if mode == "train":
            self.idx_list = data_reader.train_idx.copy()
        elif mode == "validation":
            self.idx_list = data_reader.validation_idx.copy()
        elif mode == "inference":
            self.idx_list = data_reader.inference_idx.copy()
        else:
            raise ValueError("Invalid mode")

        self.eof = None

        self.reset()

    @property
    def account(self):  # validation / inference only
        return self._account

    @property
    def previous_action(self):
        return self._previous_action

    @property
    def episode_cum_return(self):
        return self._episode_cum_return

    @account.setter  # validation / inference only
    def account(self, current_idx: int, sample: pd.Series):
        self._account[current_idx] = {
            "date": sample["date"],
            "hours": sample["hours"],
            "mins": sample["mins"],
            "close": sample["close"],
            "action": None,
            "tot_amount": None,
        }

    @previous_action.setter
    def previous_action(self, code: List):
        self._previous_action = code

    @episode_cum_return.setter
    def episode_cum_return(self, rtn: float):
        self._episode_cum_return += rtn

    def reset(self) -> None:  # epoch done
        self.eof = False
        self.previous_action = actions["NP"]
        self.episode_cum_return = 0

        if self._mode == "train":
            random.shuffle(self.idx_list)
        elif self._mode in ("validation", "inference"):
            pass
        else:
            assert False, "None Defined mode"

    def next(self) -> pd.Series:
        try:
            current_idx = self.idx_list.pop(0)
            sample = self._data_reader.sampler(current_idx)
            if self._mode in ("validation", "inference"):
                self.account = (current_idx, sample)
        except IndexError:
            self.eof = True
            sample = None

        return sample

    def done(self):  # episode done
        self.previous_action = actions["NP"]
        self.episode_cum_return = 0


if __name__ == "__main__":
    # 데이터 리더 생성
    data_reader_instance = DataReader(
        raw_filename_min="./src/local_data/raw/dax_tm3.csv",
        pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
        debug=False,
    )

    train_agent = Agent(
        mode="train",
        data_reader=data_reader_instance,
    )
    train_agent.next()

    validation_agent = Agent(
        mode="validation",
        data_reader=data_reader_instance,
    )
    validation_agent.next()

    inference_agent = Agent(
        mode="inference",
        data_reader=data_reader_instance,
    )
    inference_agent.next()

    # agent.analyse_data_to_csv("./src/local_data/intermediate/dax_anayse_data.csv")
