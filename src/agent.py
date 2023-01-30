import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from env import ENV
from util import print_c

actions = {
    "NP": [0, 0, 0, 0, 0, 0, 0],
    "BP_Init": [0, 1, 0, 0, 0, 0, 0],
    "SP_Init": [0, 0, 1, 0, 0, 0, 0],
    "BP_Hold": [0, 0, 0, 1, 0, 0, 0],
    "SP_Hold": [0, 0, 0, 0, 1, 0, 0],
    "BP_Clear": [0, 0, 0, 0, 0, 1, 0],
    "SP_Clear": [0, 0, 0, 0, 0, 0, 1],
}


class Agent(ENV):
    def __init__(
        self,
        raw_filename_min: str = None,
        pivot_filename_day: str = None,
        mode: str = None,
    ) -> None:
        super().__init__(
            raw_filename_min=raw_filename_min,
            pivot_filename_day=pivot_filename_day,
            mode=mode,
        )

        self._previous_action = actions["NP"]  # 관측 액션: T-1 시점의 액션
        self._episode_cum_return = 0
        self._account = OrderedDict()  # validation / inference 에 활용

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
        self.current_idx = self.immutable_idx.copy()
        self.eof = False
        self.previous_action = actions["NP"]
        self.episode_cum_return = 0

        if self.mode == "train":
            random.shuffle(self.current_idx)
        elif self.mode in ("validation", "inference"):
            pass
        else:
            assert False, "None Defined mode"

    def next(self) -> pd.Series:
        try:
            current_idx = self.current_idx.pop(0)
            self.sample = self._sampler(current_idx)
            if self.mode in ("validation", "inference"):
                self.account = (current_idx, self.sample)
        except IndexError:
            self.eof = True
            self.sample = None

        return self.sample

    def done(self):  # episode done
        self.previous_action = actions["NP"]
        self.episode_cum_return = 0


if __name__ == "__main__":
    train_agent = Agent(
        raw_filename_min="./src/local_data/raw/dax_tm3.csv",
        pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
        mode="train",
    )
    train_agent.next()

    validation_agent = Agent(
        raw_filename_min="./src/local_data/raw/dax_tm3.csv",
        pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
        mode="validation",
    )
    validation_agent.next()

    inference_agent = Agent(
        raw_filename_min="./src/local_data/raw/dax_tm3.csv",
        pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
        mode="inference",
    )
    inference_agent.next()

    # agent.analyse_data_to_csv("./src/local_data/intermediate/dax_anayse_data.csv")
