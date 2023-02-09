# from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import modin.pandas as pd
import numpy as np

from attributes import spread
from gen_intermediate_raw import RawDataReader
from quantile_discretizer import convert_to_index
from util import print_c


class ENV:
    def __init__(
        self,
        raw_filename_min: str = None,
        pivot_filename_day: str = None,
        mode: str = None,
        bins: int = 0,
        debug: bool = False,
    ) -> None:
        self.raw_filename_min = raw_filename_min
        self.pivot_filename_day = pivot_filename_day
        self.mode = mode  # [train | validation | inference]
        self.bins = bins
        self.debug = debug

        # 학습/추론 데이터
        self.processed_data = self._pre_process(self._read_analyse_data())
        self.n_sample = self.processed_data.shape[0]

        self.immutable_idx = self._idx_split()
        self.determinable_idx = list(self.processed_data.query("mark == True").index)
        self.current_idx, self.eof, self.sample = None, False, None
        self.candle_size = [1, 3, 5, 15, 60, 240]
        self.w_size = [9, 50, 100]

    def _read_analyse_data(self) -> pd.DataFrame:
        print_c("\n분봉/일봉/피봇(일봉) 데이터는 준비 되어 있어야 함!!!")
        raw_data = RawDataReader(
            raw_filename_min=self.raw_filename_min,
            params={"candle_size": self.candle_size, "w_size": self.w_size},
        )
        print("Raw Data Ready: OK")

        pv_data = pd.read_csv(self.pivot_filename_day)
        print("Pivot Data Ready: OK")

        analyse_data = pd.merge(raw_data.raw, pv_data, how="inner", on="date")
        analyse_data["idx"] = np.arange(analyse_data.shape[0])
        analyse_data.set_index("idx", inplace=True)
        analyse_data["date"] = pd.to_datetime(analyse_data.date)
        print("Analyse Data Ready: OK")

        if self.debug:
            analyse_data.to_csv("./assets/analyse_data.csv")
            print("Export analyse_data: OK")

        return analyse_data

    # generate the dataframe for processed data
    def _pre_process(self, analyse_data) -> pd.DataFrame:
        processed_data = pd.DataFrame()

        # 변수 생성: (종가 - 이동평균)/이동평균
        for candle_size in self.candle_size:
            for w_size in self.w_size:
                f_prc = "close"
                i_prc = f"candle{candle_size}_ma{w_size}"
                processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

        # 변수 생성: 일봉기준 (종가 - 마지노선)/마지노선
        for i_prc in [
            "10_lower_maginot",
            "10_upper_maginot",
            "20_lower_maginot",
            "20_upper_maginot",
            "50_lower_maginot",
            "50_upper_maginot",
        ]:
            f_prc = "close"
            processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

        # 변수 생성: 1분봉의 컨피던스
        for i_prc in ["high", "low"]:
            f_prc = "close"
            processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

        # 변수 생성: 각 분봉의 컨피던스, (이전 분봉의 종가 - 현재종가)/현재종가
        for candle_size in self.candle_size[1:]:
            for i_prc in ["high", "low"]:
                i_prc = f"{candle_size}mins_{i_prc}"
                f_prc = "close"
                processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)
            # 분봉의 시가 == 이전 분봉의 종가
            i_prc = f"{candle_size}mins_open"
            processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)
        # 변수 생성: 1 Forward return
        processed_data["y_rtn_close"] = analyse_data["close"].pct_change()

        # 추가 변수 타임 스템프
        processed_data["hours"] = analyse_data["hours"]
        processed_data["mins"] = analyse_data["mins"]
        processed_data["date"] = analyse_data["date"]
        processed_data["datetime"] = analyse_data["datetime"]
        processed_data["close"] = analyse_data["close"]

        # 추가
        processed_data["mark"] = analyse_data["mark"]

        return processed_data

    def _idx_split(self, infer_interval: str = "default") -> List:
        if self.mode in ("train", "validation"):
            # 70% 비율로 Seen/Un-seen 데이터 분할, 80% 비율로 학습/검증 데이터 분할
            unseen_loc = int(self.n_sample * 0.7)
            seen_loc = self.n_sample - unseen_loc
            train_valid_loc = int(seen_loc * 0.8)

            inference_idx = list(self.processed_data.iloc[unseen_loc:].index)
            train_idx = list(self.processed_data.iloc[:train_valid_loc].index)
            valid_idx = list(self.processed_data.iloc[train_valid_loc:].index)

            # Return the intersection between the data before the split index and the determinable_idx
            if self.mode == "train":
                return set.intersection(train_idx, self.determinable_idx)
            return set.intersection(valid_idx, self.determinable_idx)

        elif self.mode == "inference":
            if infer_interval == "default":
                return set.intersection(inference_idx, self.determinable_idx)
            else:
                return inference_idx  # decision every minutes

        assert False, "Invalid mode"

    def _sampler(self, current_idx: int, n_sample: int = 0, sampler: str = None):
        if sampler == "nmt_sampler_train":
            return nmt_sampler(self.processed_data, current_idx, n_sample)
        assert False, "Invalid sampler"


@lru_cache(maxsize=None)
def nmt_sampler(processed_data, current_idx: int, n_sample: int = 0):
    loc = processed_data.index.get_loc(current_idx)
    return convert_to_index(processed_data.iloc[loc - n_sample : loc + 1])
