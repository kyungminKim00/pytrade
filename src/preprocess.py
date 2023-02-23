from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import ray

from attributes import confidence_spread_candle  # confidence_candle_1,
from attributes import spread_close_ma, spread_close_maginot
from refine_raw import Refine
from util import funTime, print_c


class SequentialDataSet:
    def __init__(
        self,
        raw_filename_min: str = None,
        pivot_filename_day: str = None,
        experimental_mode: bool = True,
        debug: bool = False,
        candle_size: Tuple[int] = (1, 3, 5, 15, 60, 240),
        w_size: Tuple[int] = (9, 50, 100),
        determinable_candle: int = 3,
        offset: int = None,
    ) -> None:
        self._debug = debug
        self._candle_size = candle_size
        self._w_size = w_size
        self._determinable_candle = determinable_candle
        self._offset = offset

        self.train_idx = None
        self.train_data = None
        self.validation_idx = None
        self.validation_data = None
        self.inference_idx = None
        self.inference_data = None

        # 전체 데이터(입력 csv)의 전처리 데이터
        self.processed_data = self._pre_process(
            self._read_analyse_data(raw_filename_min, pivot_filename_day)
        )

        # 데이터 폴더 나누기
        self._total_samples = self.processed_data.shape[0]  # 학습/검증/테스트 전체 샘플 수
        self._determinable_idx = list(self.processed_data.query("mark == True").index)
        if experimental_mode:
            (
                self.train_idx,
                self.train_data,
                self.validation_idx,
                self.validation_data,
                self.inference_idx,
                self.inference_data,
            ) = self._idx_split(self._determinable_idx)
        else:  # 운영모드 or 전체 데이터가 테스트 데이터인 경우
            self.inference_idx = self._determinable_idx
            self.inference_data = self.processed_data

        # self.n_train, self.n_validation, self.n_inference = (
        #     len(self.train_idx),
        #     len(self.validation_idx),
        #     len(self.inference_idx),
        # )

    @funTime("pre_process - step 1")
    def _read_analyse_data(self, raw_filename_min, pivot_filename_day) -> pd.DataFrame:
        print_c("\n분봉/일봉/피봇(일봉) 데이터는 준비 되어 있어야 함!!!")

        raw_data = Refine(
            raw_filename_min,
            params={
                "candle_size": self._candle_size,
                "w_size": self._w_size,
                "determinable_candle": self._determinable_candle,
                "offset": self._offset,
            },
        ).raw
        print_c("Raw Data Ready: OK")

        pv_data = pd.read_csv(pivot_filename_day)
        print_c("Pivot Data Ready: OK")

        analyse_data = pd.merge(raw_data, pv_data, how="inner", on="date")
        analyse_data.set_index(raw_data.index, inplace=True)
        analyse_data["date"] = pd.to_datetime(analyse_data.date)
        print_c("Analyse Data Ready: OK")

        if self._debug:
            analyse_data.to_csv("./src/local_data/assets/analyse_data.csv")
            print_c("Export analyse_data: OK")

        return analyse_data

    @funTime("pre_process - step 2")
    def _pre_process(self, analyse_data) -> pd.DataFrame:
        # 선언문 위치 옮기지 말것
        # 공용 데이터와(mp), single cpu를 위한 Data 독립적으로 유지 하여 적은 데이터 사이즈에서 특징을 생성 하도록 함
        analyse_data_ref = ray.put(analyse_data)
        processed_data = pd.DataFrame()

        """ Parallelizes Spread Function Calculations and store the results in respective columns of Processed data
        변수 생성: (종가 - 이동평균)/이동평균
        변수 생성: 일봉기준 (종가 - 마지노선)/마지노선
        변수 생성: 1분봉의 컨피던스
        변수 생성:
            1. 각 n분봉의 컨피던스
                (현재고가 - 현재종가)/현재종가
                (현재저가 - 현재종가)/현재종가
            2.  (이전 n분봉의 종가 - 현재종가)/현재종가
                이전 n분봉의 종가 == 현재 n분봉의 시가 이므로 (이전 n분봉의 종가 - 현재종가)/현재종가 -> (현재 n분봉의 시가 - 현재종가)/현재종가 
        """
        processed_data = spread_close_ma(
            processed_data, analyse_data_ref, self._candle_size[1:], self._w_size
        )
        processed_data = spread_close_maginot(processed_data, analyse_data_ref)
        processed_data = confidence_spread_candle(
            processed_data,
            analyse_data_ref,
            self._candle_size[1:],
        )
        # not in use
        # processed_data = confidence_candle_1(processed_data, analyse_data_ref)

        # 메모리 해제
        del analyse_data_ref

        # single cpu
        processed_data["y_rtn_close"] = (
            analyse_data["close"].pct_change().shift(-1)
        )  # 1 forwards expectation
        processed_data["date"] = analyse_data["date"]
        processed_data["datetime"] = analyse_data["datetime"]
        processed_data["close"] = analyse_data["close"]
        processed_data["mark"] = analyse_data["mark"]

        # additional data
        processed_data["assets"] = "dax"

        return processed_data

    def _idx_split(self, determinable_idx) -> List:
        # 70% 비율로 Seen/Un-seen 데이터 분할, 80% 비율로 학습/검증 데이터 분할
        unseen_loc = int(self._total_samples * 0.8)
        train_validation_loc = int(unseen_loc * 0.8)

        inference_data = self.processed_data.iloc[unseen_loc:]
        inference_idx = list(inference_data.index)

        train_data = self.processed_data.iloc[:train_validation_loc]
        train_idx = list(train_data.index)

        validation_data = self.processed_data.iloc[train_validation_loc:unseen_loc]
        validation_idx = list(validation_data.index)

        # 데이터 세트 별 determinable index
        return (
            sorted(list(set.intersection(set(train_idx), set(determinable_idx)))),
            train_data,
            sorted(list(set.intersection(set(validation_idx), set(determinable_idx)))),
            validation_data,
            sorted(list(set.intersection(set(inference_idx), set(determinable_idx)))),
            inference_data,
        )

    # def __getstate__(self):
    #     state = self.__dict__.copy()

    #     # modin.pandas 피클링에 오류 있음
    #     # 텍스트로 저장 함
    #     for k, v in self.__dict__.items():
    #         try:
    #             is_pandas = isinstance(v, pd.dataframe.DataFrame)
    #             employed = "load.pandas"
    #         except AttributeError:
    #             is_pandas = isinstance(v, pd.core.frame.DataFrame)
    #             employed = "load.pandas"

    #         if is_pandas:
    #             v.to_csv(f"./src/local_data/assets/{k}.csv")
    #             state[k] = f"[{employed}]./src/local_data/assets/{k}.csv"
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    #     # modin.pandas 텍스트로 부터 객체 생성
    #     for k, v in self.__dict__.items():
    #         if isinstance(v, str):
    #             if "[load.pandas]" in v:
    #                 print_c(f"un-pickling {v}")
    #                 df = pd.read_csv(v.replace("[load.pandas]", ""))
    #                 df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
    #                 self.__dict__[k] = df
