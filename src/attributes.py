# import numpy as np
import itertools

import pandas as pd
import ray

prefix = "spd_"

# 병렬처리 적용
@ray.remote
def spread_parallel(analyse_data, f_prc, i_prc):
    return spread(analyse_data, f_prc, i_prc)


# Spread 일함수
def spread(
    df: pd.DataFrame, final_price: str = None, init_price: str = None
) -> pd.Series:
    return pd.Series(
        (df[final_price] - df[init_price]) / df[init_price],
        index=df.index,
    )


# diff 일반함수
def diff(
    df: pd.DataFrame, final_price: str = None, init_price: str = None
) -> pd.Series:
    return pd.Series(
        (df[final_price] - df[init_price]),
        index=df.index,
    )


# 변수 생성: (종가 - 이동평균)/이동평균
def spread_close_ma(
    processed_data,
    analyse_data_ref,
    _candle_size,
    _w_size,
):
    spread_futures = []
    for candle_size in _candle_size:
        for w_size in _w_size:
            f_prc = "close"
            i_prc = f"candle{candle_size}_ma{w_size}"
            spread_futures.append(
                spread_parallel.remote(analyse_data_ref, f_prc, i_prc)
            )

    for i, (candle_size, w_size) in enumerate(itertools.product(_candle_size, _w_size)):
        f_prc = "close"
        i_prc = f"candle{candle_size}_ma{w_size}"
        processed_data[f"{prefix}{f_prc}_{i_prc}"] = ray.get(spread_futures[i])

    return processed_data


# 변수 생성: 일봉기준 (종가 - 마지노선)/마지노선
def spread_close_maginot(
    processed_data,
    analyse_data_ref,
):
    spread_futures = [
        spread_parallel.remote(analyse_data_ref, "close", i_prc)
        for i_prc in [
            "10_lower_maginot",
            "10_upper_maginot",
            "20_lower_maginot",
            "20_upper_maginot",
            "50_lower_maginot",
            "50_upper_maginot",
        ]
    ]

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
        processed_data[f"{prefix}{f_prc}_{i_prc}"] = ray.get(spread_futures[i])

    return processed_data


# 변수 생성: 1분봉의 컨피던스
def confidence_candle_1(
    processed_data,
    analyse_data_ref,
):
    spread_futures = [
        spread_parallel.remote(analyse_data_ref, "close", i_prc)
        for i_prc in ["high", "low"]
    ]

    for i, i_prc in enumerate(["high", "low"]):
        f_prc = "close"
        processed_data[f"{prefix}{f_prc}_{i_prc}"] = ray.get(spread_futures[i])

    return processed_data


# 변수 생성:
# 1. 각 n분봉의 컨피던스 (현재고가 - 현재종가)/현재종가 and (현재저가 - 현재종가)/현재종가
# 2. (이전 n분봉의 종가 - 현재종가)/현재종가
# 이전 n분봉의 종가 == 현재 n분봉의 시가 이므로 (이전 n분봉의 종가 - 현재종가)/현재종가 -> (현재 n분봉의 시가 - 현재종가)/현재종가
def confidence_spread_candle(
    processed_data,
    analyse_data_ref,
    _candle_size,
):
    spread_futures = []
    for candle_size in _candle_size:
        for i_prc in ["high", "low", "open"]:
            spread_futures.append(
                spread_parallel.remote(
                    analyse_data_ref, "close", f"{candle_size}mins_{i_prc}"
                )
            )

    for i, (candle_size, w_size) in enumerate(
        itertools.product(_candle_size, ["high", "low", "open"])
    ):
        i_prc = f"{candle_size}mins_{w_size}"
        f_prc = "close"
        processed_data[f"{prefix}{f_prc}_{i_prc}"] = ray.get(spread_futures[i])

    return processed_data
