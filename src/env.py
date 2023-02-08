# from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import modin.pandas as pd

from gen_intermediate_raw import RawDataReader
from util import print_c
from attributes import spread


class ENV:
    def __init__(
        self,
        raw_filename_min: str = None,
        pivot_filename_day: str = None,
        mode: str = None,
        bins: int = 0,
    ) -> None:
        self.raw_filename_min = raw_filename_min
        self.pivot_filename_day = pivot_filename_day
        self.mode = mode  # [train | validation | inference]
        self.bins = bins

        self.analyse_data = self._read_analyse_data()
        self.processed_data = self._pre_process()
        self.n_sample = self.processed_data.shape[0]

        self.immutable_idx = self._splite_data()
        self.current_idx, self.eof, self.sample = None, False, None
        self.sample_set = {}
        self.candle_size = [1, 3, 5, 15]
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
        return analyse_data

    def analyse_data_to_csv(self, outfile: str = None) -> None:
        self.analyse_data.to_csv(outfile)

    def _pre_process(self) -> pd.DataFrame:
        processed_data = pd.DataFrame()

        # 변수 생성: (종가 - 이동평균)/이동평균
        for candle_size in self.candle_size:
            for w_size in self.w_size:
                f_prc = "close"
                i_prc = f"candle{candle_size}_ma{w_size}"
                processed_data[f"{f_prc}_{i_prc}"] = spread(
                    self.analyse_data, f_prc, i_prc
                )

        # 변수 생성: (종가 - 마지노선)/마지노선
        for i_prc in [
            "10_lower_maginot",
            "10_upper_maginot",
            "20_lower_maginot",
            "20_upper_maginot",
            "50_lower_maginot",
            "50_upper_maginot",
        ]:
            f_prc = "close"
            processed_data[f"{f_prc}_{i_prc}"] = spread(self.analyse_data, f_prc, i_prc)

        # 변수 생성: 1분봉의 컨피던스
        for i_prc in ["high", "low"]:
            f_prc = "close"
            processed_data[f"{f_prc}_{i_prc}"] = spread(self.analyse_data, f_prc, i_prc)

        # 변수 생성: 각 분봉의 컨피던스, (이전 분봉의 종가 - 현재종가)/현재종가
        for candle_size in self.candle_size[1:]:
            for i_prc in ["high", "low"]:
                i_prc = f"{candle_size}mins_{i_prc}"
                f_prc = "close"
                processed_data[f"{f_prc}_{i_prc}"] = spread(
                    self.analyse_data, f_prc, i_prc
                )
            # 분봉의 시가 == 이전 분봉의 종가
            i_prc = f"{candle_size}mins_open"
            processed_data[f"{f_prc}_{i_prc}"] = spread(self.analyse_data, f_prc, i_prc)
        # 변수 생성: 1 Forward return
        processed_data["y_rtn_close"] = self.analyse_data["close"].pct_change()

        # 추가 변수 타임 스템프
        processed_data["hours"] = self.analyse_data["hours"]
        processed_data["mins"] = self.analyse_data["mins"]
        processed_data["date"] = self.analyse_data["date"]
        processed_data["datetime"] = self.analyse_data["datetime"]
        processed_data["close"] = self.analyse_data["close"]

        return processed_data

    def _splite_data(self) -> List:
        """
        train/validation: dax index
        test: Nasdaq(IXIC) index
        문제점: 지수는 거래량이 1일후에 나오는 것은 상관없으나, 지수를 실시간 매매할수 없음
        문제점: 트레이더블 지수 추종 상품과 지수는 가격지표가 다르기 때문에 직접 모형을 적용 할 수 없음
        문제점: 트레이더블 지수 추종 상품의 거래량을 믿어도 될 지 확신이 없음
        위의 세가지 이슈는 운영시 고려되어야 함
        """
        print("여기 부터 시작")

        if self.mode in ("train", "validation"):  # train/validation
            splite_idx = int(self.n_sample * 0.7)
            if self.mode == "train":
                return list(self.processed_data.index[:splite_idx])
            return list(self.processed_data.index[splite_idx:])
        elif self.mode == "inference":  # inference
            return list(self.processed_data.index)
        return None

    def _sampler(self, current_idx: int) -> np:
        try:
            sample = self.sample_set[current_idx]
        except KeyError:
            self.sample_set[current_idx] = self.processed_data.iloc[current_idx]

            sample = self.sample_set[current_idx]
        return sample

    # @abstractmethod
    # def reset(self) -> None:
    #     pass

    # @abstractmethod
    # def next(self) -> None:
    #     pass

    # @abstractmethod
    # def done(self) -> None:
    #     pass
