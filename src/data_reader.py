# from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import modin.pandas as pd
import numpy as np

from attributes import spread, spread_parallel
from quantile_discretizer import encode
from refine_raw import Refine
from util import funTime, print_c


class DataReader:
    def __init__(
        self,
        raw_filename_min: str = None,
        pivot_filename_day: str = None,
        experimental_mode: bool = True,
        debug: bool = False,
        candle_size: Tuple[int] = (1, 3, 5, 15, 60, 240),
        w_size: Tuple[int] = (9, 50, 100),
    ) -> None:

        self._debug = debug
        self._candle_size = candle_size
        self._w_size = w_size

        # 전체 데이터(입력 csv)의 전처리 데이터
        self.processed_data = self._pre_process(
            self._read_analyse_data(raw_filename_min, pivot_filename_day)
        )

        # 데이터 폴더 나누기
        self.total_samples = self.processed_data.shape[0]  # 학습/검증/테스트 전체 샘플 수
        self._determinable_idx = list(self.processed_data.query("mark == True").index)
        if experimental_mode:
            self.train_idx, self.validation_idx, self.inference_idx = self._idx_split(
                self._determinable_idx
            )
        else:  # 운영모드 or 전체 데이터가 테스트 데이터인 경우
            self.inference_idx = self._determinable_idx

        self.n_train, self.n_validation, self.n_inference = (
            len(self.train_idx),
            len(self.validation_idx),
            len(self.inference_idx),
        )

    @funTime("pre_process - step 1")
    def _read_analyse_data(self, raw_filename_min, pivot_filename_day) -> pd.DataFrame:
        print_c("\n분봉/일봉/피봇(일봉) 데이터는 준비 되어 있어야 함!!!")
        raw_data = Refine(
            raw_filename_min,
            params={"candle_size": self._candle_size, "w_size": self._w_size},
        ).raw
        print("Raw Data Ready: OK")

        pv_data = pd.read_csv(pivot_filename_day)
        print("Pivot Data Ready: OK")

        analyse_data = pd.merge(raw_data, pv_data, how="inner", on="date")
        analyse_data.set_index(raw_data.index, inplace=True)
        analyse_data["date"] = pd.to_datetime(analyse_data.date)
        print("Analyse Data Ready: OK")

        if self._debug:
            analyse_data.to_csv("./assets/analyse_data.csv")
            print("Export analyse_data: OK")

        return analyse_data

    # @funTime("pre_process - step 2")
    # def _pre_process(self, analyse_data) -> pd.DataFrame:
    #     processed_data = pd.DataFrame()

    #     # 변수 생성: (종가 - 이동평균)/이동평균
    #     for candle_size in self._candle_size:
    #         for w_size in self._w_size:
    #             f_prc = "close"
    #             i_prc = f"candle{candle_size}_ma{w_size}"
    #             processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

    #     # 변수 생성: 일봉기준 (종가 - 마지노선)/마지노선
    #     for i_prc in [
    #         "10_lower_maginot",
    #         "10_upper_maginot",
    #         "20_lower_maginot",
    #         "20_upper_maginot",
    #         "50_lower_maginot",
    #         "50_upper_maginot",
    #     ]:
    #         f_prc = "close"
    #         processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

    #     # 변수 생성: 1분봉의 컨피던스
    #     for i_prc in ["high", "low"]:
    #         f_prc = "close"
    #         processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

    #     # 변수 생성: 각 분봉의 컨피던스, (이전 분봉의 종가 - 현재종가)/현재종가
    #     for candle_size in self._candle_size[1:]:
    #         for i_prc in ["high", "low"]:
    #             i_prc = f"{candle_size}mins_{i_prc}"
    #             f_prc = "close"
    #             processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)
    #         # 분봉의 시가 == 이전 분봉의 종가
    #         i_prc = f"{candle_size}mins_open"
    #         processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)
    #     # 변수 생성: 1 Forward return
    #     processed_data["y_rtn_close"] = analyse_data["close"].pct_change()

    #     # 추가 변수 타임 스템프
    #     processed_data["hours"] = analyse_data["hours"]
    #     processed_data["mins"] = analyse_data["mins"]
    #     processed_data["date"] = analyse_data["date"]
    #     processed_data["datetime"] = analyse_data["datetime"]
    #     processed_data["close"] = analyse_data["close"]

    #     # 추가
    #     processed_data["mark"] = analyse_data["mark"]

    #     return processed_data

    @funTime("pre_process - step 2")
    def _pre_process(self, analyse_data) -> pd.DataFrame:
        import itertools

        import ray

        processed_data = pd.DataFrame()

        # 변수 생성: (종가 - 이동평균)/이동평균
        spread_futures = []
        for candle_size in self._candle_size:
            for w_size in self._w_size:
                f_prc = "close"
                i_prc = f"candle{candle_size}_ma{w_size}"
                spread_futures.append(
                    spread_parallel.remote(analyse_data, f_prc, i_prc)
                )

        for i, (candle_size, w_size) in enumerate(
            itertools.product(self._candle_size, self._w_size)
        ):
            f_prc = "close"
            i_prc = f"candle{candle_size}_ma{w_size}"
            processed_data[f"{f_prc}_{i_prc}"] = ray.get(spread_futures[i])

        # 변수 생성: 일봉기준 (종가 - 마지노선)/마지노선
        spread_futures = []
        for i_prc in [
            "10_lower_maginot",
            "10_upper_maginot",
            "20_lower_maginot",
            "20_upper_maginot",
            "50_lower_maginot",
            "50_upper_maginot",
        ]:
            f_prc = "close"
            spread_futures.append(spread_parallel.remote(analyse_data, f_prc, i_prc))

        for i, i_prc in enumerate(
            [
                "10_lower_maginot",
                "10_upper_maginot",
                "20_lower_maginot",
                "20_upper_maginot",
                "50_lower_maginot",
                "50_upper_maginot",
            ]
        ):
            f_prc = "close"
            processed_data[f"{f_prc}_{i_prc}"] = ray.get(spread_futures[i])

        # 변수 생성: 1분봉의 컨피던스
        for i_prc in ["high", "low"]:
            f_prc = "close"
            processed_data[f"{f_prc}_{i_prc}"] = spread(analyse_data, f_prc, i_prc)

        # 변수 생성: 각 분봉의 컨피던스, (이전 분봉의 종가 - 현재종가)/현재종가
        for candle_size in self._candle_size[1:]:
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

    def _idx_split(self, determinable_idx) -> List:
        # 70% 비율로 Seen/Un-seen 데이터 분할, 80% 비율로 학습/검증 데이터 분할
        unseen_loc = int(self.total_samples * 0.7)
        seen_loc = self.total_samples - unseen_loc
        train_valid_loc = int(seen_loc * 0.8)

        inference_idx = list(self.processed_data.iloc[unseen_loc:].index)
        train_idx = list(self.processed_data.iloc[:train_valid_loc].index)
        valid_idx = list(self.processed_data.iloc[train_valid_loc:].index)

        # 데이터 세트 별 determinable index
        return (
            sorted(list(set.intersection(set(train_idx), set(determinable_idx)))),
            sorted(list(set.intersection(set(valid_idx), set(determinable_idx)))),
            sorted(list(set.intersection(set(inference_idx), set(determinable_idx)))),
        )

    def sampler(self, query_idx: int, n_sample: int = 0, sampler: str = None):
        if sampler == "nmt_sampler_train":
            return _nmt_sampler(self.processed_data, query_idx, n_sample)
        assert False, "Invalid sampler"


@lru_cache(maxsize=None)
def _nmt_sampler(processed_data, query_idx: int, n_pre_trajectory: int = 0):
    loc = processed_data.index.get_loc(query_idx)
    return encode(processed_data.iloc[loc - n_pre_trajectory : loc + 1])
