import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import bottleneck as bn
import modin.pandas as pd
import numpy as np
import psutil
import ray

from util import print_c


class Refine:
    def __init__(
        self,
        raw_filename_min: str = None,
        params: Dict[str, int] = None,
    ) -> None:
        # init ray
        ray.shutdown()
        ray.init(num_cpus=psutil.cpu_count(logical=False))

        # load data from csv
        r_pd = pd.read_csv(
            raw_filename_min, sep=",", dtype={"date": object, "time": object}
        )

        self._candle_size = params["candle_size"]
        self._w_size = params["w_size"]
        self._raw = self._refine_process(r_pd)

    @property
    def raw(self):
        return self._raw

    def _refine_process(self, r_pd: pd.DataFrame) -> pd.DataFrame:
        # 추후 삭제 - 코드개발시 데이터 사이즈 줄이기 위해 존재하는 코드
        r_pd = r_pd[-35000:]
        print_c("Reduce the size of raw data - Remove this section")

        # generate datetime
        # r_pd["hours"] = r_pd["time"].str[:-2]
        # r_pd["mins"] = r_pd["time"].str[-2:]
        # r_pd["datetime"] = pd.to_datetime(
        #     r_pd["date"] + " " + r_pd["hours"] + ":" + r_pd["mins"]
        # )
        r_pd["datetime"] = pd.to_datetime(
            r_pd["date"] + " " + r_pd["time"].str[:-2] + ":" + r_pd["time"].str[-2:]
        )
        # 인덱스 설정
        r_pd.set_index("datetime", inplace=True, drop=False)

        # 리파인 raw 1분갱신 캔들가격과 캔들이평 추가
        r_pd = self._generate_candle_price(r_pd)

        # 3분봉 기준 의사결정 가능
        r_pd = self._mark_determinable(r_pd)

        return r_pd

    def _mark_determinable(
        self, r_pd: pd.DataFrame, determinable_candle: int = 3
    ) -> pd.DataFrame:
        determinable_datetime = r_pd["datetime"].dt.floor(f"{determinable_candle}T")
        r_pd["mark"] = r_pd["datetime"] == determinable_datetime
        print_c("Mark determinable date: Done")
        return r_pd

    def _generate_candle_price(self, r_pd: pd.DataFrame) -> pd.DataFrame:
        # Tip: 캔들의 시작에 데이터 수집이 유리 함 (예. 1분 1초)

        # case: candle_size = 5min, moving_averages = 9
        for candle_size in self._candle_size:
            identifier = f"candle{candle_size}_datetime"
            r_pd[identifier] = r_pd["datetime"].dt.floor(f"{candle_size}T")

            # 캔들에 대응 하는 매분 업데이트 데이터 (예. 3분봉의 매분 업데이트 데이터)
            if candle_size > 1:
                # (시가) 캔들의 시초가 (캔들의 시초가는 해당 시간내 변하지 않는 값)
                candle_open = r_pd.groupby(identifier)["open"].apply(
                    lambda x: x.iloc[0]
                )
                candle_open.name = f"{candle_size}mins_open"
                r_pd = r_pd.join(candle_open, on=identifier, how="outer")

                # (고가) 매분 갱신되는 캔들의 데이터 (최대값 갱신)
                candle_high = r_pd.groupby(identifier)["high"].apply(
                    lambda x: x.cummax()
                )
                candle_high.name = f"{candle_size}mins_high"
                r_pd = r_pd.join(candle_high, how="inner")

                # (저가) 매분 갱신되는 캔들의 데이터 (최소값 갱신)
                candle_low = r_pd.groupby(identifier)["low"].apply(lambda x: x.cummin())
                candle_low.index.name = identifier
                candle_low.name = f"{candle_size}mins_low"
                r_pd = r_pd.join(candle_low, how="inner")

                # (종가) 매분 갱신되는 캔들의 데이터 (최신 종가가 캔들의 종가)
                r_pd[f"{candle_size}mins_close"] = r_pd["close"]

            # 각 캔들에 대응하는 이동평균 데이터
            for w_size in self._w_size:
                g_candle = f"candle{str(candle_size)}_ma{str(w_size)}"
                data = r_pd.groupby(identifier)["close"].apply(lambda x: x.iloc[-1])
                moving_avg = data.rolling(window=w_size).mean()
                moving_avg.name = g_candle
                moving_avg = moving_avg.to_frame()
                r_pd = r_pd.join(moving_avg, on=identifier, how="outer")

        r_pd = r_pd.dropna(axis=0)
        print_c("Refine candle prices: Done")
        return r_pd

    def pd_formatting(self, data):
        data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volumn",
                "Adj Close": "adj close",
            },
            inplace=True,
        )
        data["date"] = pd.to_datetime(data.date)
        data["date"] = data["date"].dt.strftime("%m/%d/%Y")
        return data
