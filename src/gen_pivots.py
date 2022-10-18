from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import ray


@ray.remote
def mp_log_transformation(values: np, col_name: str = None) -> List:
    return f"{col_name}_log", np.log(values)


class Aggregator:
    def __init__(self, data: pd.DataFrame) -> None:
        # init ray
        ray.shutdown()
        ray.init(num_cpus=psutil.cpu_count(logical=False))

        self.data = data
        self.alpha, self.beta = 20, 20
        self.maginot_memory_fix = 500
        self.omega_fix = 0.01
        self.max_search_candidate_prices_fix = 60

        self.maginot_bins = [10, 20, 50]
        for m_bin in self.maginot_bins:
            self.upper_miginot_stack, self.lower_miginot_stack = [], []
            idx_pivot = []
            for idx in range(0, self.data.shape[0], 1):
                if idx < m_bin:
                    lower_maginot, upper_maginot = np.NaN, np.NaN
                else:
                    (
                        lower_maginot,
                        upper_maginot,
                    ) = self.retrive_maginot(self.data[idx - m_bin + 1 : idx + 1])
                idx_pivot.append(
                    [
                        idx,
                        lower_maginot,
                        upper_maginot,
                    ]
                )
            aa = pd.DataFrame(
                data=np.array(idx_pivot)[:, 1:],
                columns=[f"{m_bin}_lower_maginot", f"{m_bin}_upper_maginot"],
                index=np.array(idx_pivot)[:, 0],
            )
            self.data = pd.concat([self.data, aa], axis=1, join="inner")

        # drop null and convert formatting
        self.data = self.data.dropna(axis=0)
        self.data = self._log_transformation(self.data)
        self.pd_formatting()

    def pd_formatting(self) -> None:
        self.data.rename(
            columns={
                "Date": "date",
                "Open": "open_day",
                "High": "high_day",
                "Low": "low_day",
                "Close": "close_day",
                "Volume": "volumn_day",
                "Adj Close": "adj_close_day",
            },
            inplace=True,
        )
        self.data["date"] = pd.to_datetime(self.data.date)
        self.data["date"] = self.data["date"].dt.strftime("%m/%d/%Y")

    def _log_transformation(self, r_pd: pd.DataFrame) -> pd.DataFrame:
        res = ray.get(
            [
                mp_log_transformation.remote(r_pd[col_name].values, col_name)
                for col_name in [
                    "10_lower_maginot",
                    "10_upper_maginot",
                    "20_lower_maginot",
                    "20_upper_maginot",
                    "50_lower_maginot",
                    "50_upper_maginot",
                ]
            ]
        )
        for k, v in res:
            r_pd[k] = v
        print("Log Transformation: Done")
        return r_pd

    def historical_g_maginot(self, stack: List) -> List[float]:
        miginot = []
        for it in stack:
            tmp_cnt = stack.count(it)
            miginot.append([it, tmp_cnt])
        val = sorted(miginot, key=lambda tmp_cnt: tmp_cnt[1], reverse=True)[0][0]
        return val

    def get_maginot(
        self,
        price_scoring: np.array,
        p1: float = None,
        section: str = None,
        prices: np.array = None,
    ) -> float:
        PRICE, VOLUME = 0, 1

        if section == "lower_maginot":
            maginot = price_scoring[
                np.argwhere(price_scoring[:, PRICE] <= p1).squeeze(), :
            ]
            maginot = np.min(prices[:, PRICE]) if len(maginot) == 0 else maginot
        elif section == "upper_maginot":
            maginot = price_scoring[
                np.argwhere(price_scoring[:, PRICE] >= p1).squeeze(), :
            ]
            maginot = np.max(prices[:, PRICE]) if len(maginot) == 0 else maginot

        # if type(maginot) == np.float64:
        if isinstance(maginot, np.float64):
            return maginot
        else:
            search_bound = self.max_search_candidate_prices_fix
            if maginot.ndim > 1:
                if len(maginot) < search_bound:
                    search_bound = len(maginot)
                if section == "lower_maginot":
                    maginot = maginot[-search_bound:]
                if section == "upper_maginot":
                    maginot = maginot[:search_bound]

                maginot = np.array(
                    sorted(maginot, key=lambda volume: volume[VOLUME], reverse=True)
                )[0][PRICE]
            else:
                # maginot = maginot[:, PRICE]
                maginot = maginot[PRICE]

            return maginot

    def collect_prices(self, h_data: np.array) -> np.array:

        decision_table = np.abs(h_data["High"] - h_data["Close"]) > np.abs(
            h_data["Low"] - h_data["Close"]
        )
        aa = np.expand_dims(
            np.where(decision_table, h_data["High"], h_data["Low"]), axis=0
        )
        bb = np.expand_dims(h_data["Volume"], axis=0)
        p_data = np.append(aa.T, bb.T, axis=1).tolist()

        return np.array(sorted(p_data, key=lambda price: price[0]))

    def quntising_price_scoring(self, prices: np.array) -> np.array:
        # quntising and price scoring
        price_scoring = []
        percentile = np.percentile(
            prices[:, 0], np.arange(100), interpolation="nearest"
        )

        percentile = np.arange(
            np.min(prices[:, 0]),
            np.max(prices[:, 0]),
            (np.max(prices[:, 0]) - np.min(prices[:, 0])) * self.omega_fix,
        )
        percentile[-1] = np.max(prices[:, 0])
        # for idx in range(len(percentile)):
        for idx, _ in enumerate(percentile):
            if idx == 0:
                pass
            else:
                min_bound = percentile[idx - 1]
                max_bound = percentile[idx]
                quntile = prices[
                    np.argwhere(
                        (prices[:, 0] >= min_bound) & (prices[:, 0] <= max_bound)
                    ).squeeze(),
                    :,
                ]
                if len(quntile) == 0:
                    total_volume = 0
                else:
                    if quntile.ndim == 1:
                        total_volume = np.sum(quntile[-1])
                    else:
                        total_volume = np.sum(quntile[:, -1])
                price_scoring.append([min_bound, total_volume])
        return price_scoring

    def insert_stack(self, stack: List[float], i_maginot: float):
        if len(stack) > self.maginot_memory_fix:
            stack.pop(0)
        stack.append(i_maginot)

    def retrive_maginot(self, h_data: np.array) -> List:
        prices = self.collect_prices(h_data)
        price_scoring = self.quntising_price_scoring(prices)
        price_scoring = np.array(sorted(price_scoring, key=lambda price: price[0]))

        # select lower_maginot, upper_maginot
        lower_maginot = self.get_maginot(
            price_scoring,
            p1=h_data.iloc[-1]["Close"],
            section="lower_maginot",
            prices=prices,
        )
        upper_maginot = self.get_maginot(
            price_scoring,
            p1=h_data.iloc[-1]["Close"],
            section="upper_maginot",
            prices=prices,
        )

        for _stack, _item in [
            [self.lower_miginot_stack, lower_maginot],
            [self.upper_miginot_stack, upper_maginot],
        ]:
            self.insert_stack(_stack, _item)

        base_lower_maginot = self.historical_g_maginot(
            self.lower_miginot_stack[-self.alpha :]
        )
        base_upper_maginot = self.historical_g_maginot(
            self.upper_miginot_stack[-self.beta :]
        )

        if base_upper_maginot < base_lower_maginot:
            base_lower_maginot, base_upper_maginot = (
                base_upper_maginot,
                base_lower_maginot,
            )

        return base_lower_maginot, base_upper_maginot
