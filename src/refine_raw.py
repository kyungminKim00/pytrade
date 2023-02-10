from typing import Any, Dict, List, Optional, Tuple, Union

import modin.pandas as pd
import numpy as np
import psutil
import ray

from util import print_c

# pivot_col = [
#     "candle1_ma9",
#     "candle1_ma9",
#     "candle1_ma50",
#     "candle1_ma100",
#     "candle3_ma9",
#     "candle3_ma50",
#     "candle3_ma100",
#     "candle5_ma9",
#     "candle5_ma50",
#     "candle5_ma100",
#     "candle15_ma9",
#     "candle15_ma50",
#     "candle15_ma100",
#     "10_lower_maginot",
#     "10_upper_maginot",
#     "20_lower_maginot",
#     "20_upper_maginot",
#     "50_lower_maginot",
#     "50_upper_maginot",
# ]
# sensingval_col = [
#     "open",
#     "high",
#     "low",
#     "close",
#     "3mins_open",
#     "3mins_high",
#     "3mins_low",
#     "3mins_close",
#     "5mins_open",
#     "5mins_high",
#     "5mins_low",
#     "5mins_close",
#     "15mins_open",
#     "15mins_high",
#     "15mins_low",
#     "15mins_close",
# ]


# @ray.remote
# def mp_log_transformation(values: np, col_name: str = None) -> List:
#     return f"{col_name}_log", np.log(values)


# @ray.remote
# def mp_moving_averages(values: np, w_size: int = None, candle_size: int = None) -> List:
#     return f"candle{str(candle_size)}_ma{str(w_size)}", bn.move_mean(
#         values, window=w_size * candle_size
#     )


# @ray.remote
# def mp_gen_candles2(
#     partial_pd: pd.DataFrame,
#     min_range: List,
# ) -> List:
#     candle_time_interval = None
#     first_cursor = None
#     last_cursor = None

#     # declare last_cursor
#     partial_pd_idxs = partial_pd.index
#     # partial_pd_idxs = sorted(list(partial_pd.keys()))
#     last_cursor = partial_pd_idxs[-1]

#     # # find candle_time_interval
#     x_mins = int(partial_pd.loc[last_cursor]["mins"])
#     # x_mins = int(partial_pd[last_cursor]["mins"])

#     for v in min_range:
#         if v[0] <= x_mins < v[1]:
#             candle_time_interval = v
#     assert candle_time_interval is not None, "Data Errors!!!"

#     # find first_cursor
#     for t_idx in partial_pd_idxs:
#         x_mins = int(partial_pd.loc[t_idx]["mins"])
#         # x_mins = int(partial_pd[last_cursor]["mins"])
#         if candle_time_interval[0] <= x_mins < candle_time_interval[1]:
#             first_cursor = t_idx
#             break
#     assert first_cursor is not None, "Data Errors!!!"
#
# return ohlc_record(partial_pd, first_cursor, last_cursor)


# def ohlc_record(partial_pd, first_cursor, last_cursor):
#     candle_open = partial_pd.loc[first_cursor]["open"]
#     candle_high = partial_pd.loc[first_cursor:]["high"].max()
#     candle_low = partial_pd.loc[first_cursor:]["low"].min()
#     candle_close = partial_pd.loc[last_cursor]["close"]

# candle_open = partial_pd[first_cursor]["open"]
# candle_high = partial_pd[first_cursor:]["high"].max()
# candle_low = partial_pd[first_cursor:]["low"].min()
# candle_close = partial_pd[last_cursor]["close"]

# data = [last_cursor, candle_open, candle_high, candle_low, candle_close]
# return data


# @ray.remote
# def mp_gen_candles(values: np, max_time: int, sub_window_size: int) -> np:
#     stride_size = sub_window_size
#     sub_windows = (
#         np.expand_dims(np.arange(sub_window_size), 0)
#         + np.expand_dims(np.arange(max_time + 1), 0).T
#     )
#     sub_windows = sub_windows[::stride_size]

#     add_dummy = None
#     eov = sub_windows[-1][-1]  # the end of value
#     if (max_time - 1) < eov:
#         add_dummy = True
#         eor = sub_windows[-1]  # the end of rows
#         sub_windows = sub_windows[:-1]
#     else:
#         add_dummy = False

#     # open_val = np.array(
#     #     [
#     #         it
#     #         for it in list(
#     #             values[sub_windows][:, 0, 0] for roof in range(sub_window_size)
#     #         )
#     #     ]
#     # ).T.flatten()
#     open_val = np.array(
#         list(list((values[sub_windows][:, 0, 0] for roof in range(sub_window_size))))
#     ).T.flatten()

#     max_val = np.array(
#         [
#             it
#             for it in list(map(np.max, values[sub_windows]))
#             for roof in range(sub_window_size)
#         ]
#     ).T.flatten()
#     min_val = np.array(
#         [
#             it
#             for it in list(map(np.min, values[sub_windows]))
#             for roof in range(sub_window_size)
#         ]
#     ).T.flatten()
#     close_val = np.array(
#         [
#             it
#             for it in list(values[sub_windows][:, -1, -1])
#             for roof in range(sub_window_size)
#         ]
#     ).T.flatten()
#     # data = np.array([it for it in [open_val, max_val, min_val, close_val]]).T
#     data = np.array([open_val, max_val, min_val, close_val]).T

#     if add_dummy:
#         dummy = values[eor[eor < max_time]]
#         data = np.append(data, dummy, axis=0)

# return f"{sub_window_size}mins", data


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
        r_pd = r_pd[-2000:]
        print_c("Reduce the size of raw data - Remove this section")

        # hours and minutes
        r_pd["hours"] = r_pd["time"].str[:-2]
        r_pd["mins"] = r_pd["time"].str[-2:]

        # generate datetime
        r_pd["datetime"] = pd.to_datetime(
            r_pd["date"] + " " + r_pd["hours"] + ":" + r_pd["mins"]
        )

        # 인덱스 설정
        r_pd.set_index("datetime", inplace=True, drop=False)

        # 리파인 raw 1분갱신 캔들가격과 캔들이평 추가
        r_pd = self._generate_candle_price(r_pd)

        # 3분봉 기준 의사결정 가능
        r_pd = self._mark_determinable(r_pd)

        # option. 실제 쓰이는 분봉 데이터 생성을 위해 00분에 맞추어서 데이터 재조정
        r_pd = self._calibrate_min(r_pd)

        # # 분봉 데이터 생성
        # r_pd = self._gen_candles2(r_pd)

        # # 로그 변환
        # r_pd = self._log_transformation(r_pd)

        return r_pd

    # def _gen_candles(self, r_pd: pd.DataFrame) -> pd.DataFrame:
    #     values = r_pd[["open", "high", "low", "close"]].values

    #     max_time = values.shape[0]
    #     res = ray.get(
    #         [
    #             mp_gen_candles.remote(values, max_time, sub_window_size)
    #             for sub_window_size in self._candle_size[1:]
    #         ]
    #     )
    #     for k, v in res:
    #         r_pd = pd.concat(
    #             [
    #                 r_pd,
    #                 pd.DataFrame(
    #                     data=v,
    #                     columns=[f"{k}_open", f"{k}_high", f"{k}_low", f"{k}_close"],
    #                     index=r_pd.index,
    #                 ),
    #             ],
    #             axis=1,
    #             join="inner",
    #         )
    #     print("Gen Candles: Done")
    #     return r_pd

    # def _gen_candles2(self, r_pd: pd.DataFrame) -> pd.DataFrame:
    #     r_pd.to_csv("./src/local_data/raw/gen_candles_debug_source.csv")
    #     for candle_size in self._candle_size[1:]:
    #         min_range = list(get_min_range(candle_size).values())

    #         # 3/5/15 분봉의 시고저종 산출
    #         res = ray.get(
    #             [
    #                 mp_gen_candles2.remote(
    #                     r_pd.loc[cursor - candle_size : cursor],
    #                     min_range,
    #                 )
    #                 for cursor in list(
    #                     r_pd.index[candle_size:]
    #                 )  # cursor 포함되는 데이터여야 함 (인덱스 이므로)
    #             ]
    #         )

    #         # 계산된 3/5/15 분봉의 시고저종을 원본데이터 프레임에 추가
    #         identifier = f"{candle_size}mins"
    #         res_dict = {
    #             idx: {
    #                 f"{identifier}_open": o,
    #                 f"{identifier}_high": h,
    #                 f"{identifier}_low": l,
    #                 f"{identifier}_close": c,
    #             }
    #             for idx, o, h, l, c in res
    #         }
    #         res_pd = pd.DataFrame.from_dict(res_dict, orient="index")
    #         r_pd = pd.concat(
    #             [
    #                 r_pd,
    #                 res_pd,
    #             ],
    #             axis=1,
    #             join="inner",
    #         )
    #         print(f"{identifier} 캔들이 생성 되었습니다")

    #     r_pd.to_csv("./src/local_data/raw/gen_candles_debug_target.csv")

    #     return r_pd

    def _calibrate_min(self, r_pd: pd.DataFrame) -> pd.DataFrame:
        first_index = r_pd.index[0]
        first_00min_index = r_pd["mins"][r_pd["mins"] == "00"].index[0]
        print("Calibrate data: Done")
        return r_pd.drop(labels=range(first_index, first_00min_index), axis=0)

    def _mark_determinable(
        self, r_pd: pd.DataFrame, determinable_candle: int = 3
    ) -> pd.DataFrame:
        r_pd["mark"] = np.where(
            r_pd["datetime"] % determinable_candle == 0, True, False
        )
        print("Mark determinable date: Done")
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
                # candle_high = r_pd.groupby(identifier)["high"].cummax()

                # candle_high = r_pd.groupby(identifier)["high"].apply(
                #     lambda x: x.cummax()
                # )

                candle_high = r_pd.groupby(identifier).apply(
                    lambda x: x["high"].cummax()
                )
                candle_high.reset_index(identifier, drop=True, inplace=True)
                candle_high.rename(
                    columns={"high": f"{candle_size}mins_high"}, inplace=True
                )
                r_pd = r_pd.join(candle_high, how="inner")

                # (저가) 매분 갱신되는 캔들의 데이터 (최소값 갱신)
                # candle_low = r_pd.groupby(identifier)["low"].cummin()
                # candle_low.index.name = identifier
                # candle_low.name = f"{candle_size}mins_low"
                # r_pd = r_pd.join(candle_low, on=identifier, how="inner")
                candle_low = r_pd.groupby(identifier).apply(lambda x: x["low"].cummin())
                candle_low.reset_index(identifier, drop=True, inplace=True)
                candle_low.rename(
                    columns={"low": f"{candle_size}mins_low"}, inplace=True
                )
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

        # # old version
        # res = ray.get(
        #     [
        #         mp_moving_averages.remote(r_pd["close"].values, w_size, candle_size)
        #         for candle_size in self._candle_size
        #         for w_size in self._w_size
        #     ]
        # )
        # for k, v in res:
        #     r_pd[k] = v

        r_pd = r_pd.dropna(axis=0)
        print("Refine candle prices: Done")
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

    # def _log_transformation(self, r_pd: pd.DataFrame) -> pd.DataFrame:
    #     res = ray.get(
    #         [
    #             mp_log_transformation.remote(r_pd[col_name].values, col_name)
    #             for col_name in sensingval_col
    #         ]
    #     )
    #     for k, v in res:
    #         r_pd[k] = v
    #     print("Log Transformation: Done")
    #     return r_pd