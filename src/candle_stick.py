from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import psutil as ps
import ray


class CandleStick:
    r"""
    Class for calculating similar candle stick patterns.

    Args:
        feed_dict: A dictionary containing data for candle stick chart("Open", "High", "Low", "Close").
        bins: The number of data points to use for similarity calculation.
        fwd: The number of forward data points to use for probability calculation.
        fwd_plot_size: The size of the plot for the forward data points.
        max_samples: The maximum number of similar dates to return.

    Examples:
        >>> cs = CandleStick(feed_dict=data, bins=3, fwd=5, fwd_plot_size=15, max_samples=100)
        >>> similar_dates = cs.similar_candle(-1)
        >>> print(f"similar_dates: {similar_dates}")
    """

    def __init__(
        self,
        feed_dict: Dict[str, Dict[str, float]],
        bins: int = 3,
        fwd: int = 5,
        fwd_plot_size: int = 15,
        max_samples: int = 100,
    ) -> None:
        self._bins: int = bins
        self._fwd: int = fwd
        self._fwd_plot_size: int = fwd_plot_size
        self._max_samples: int = max_samples

        self._lengths: List[int] = []
        for k, v in feed_dict.items():
            assert k in [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
            ], "Input dictionary column error"
            assert isinstance(k, str) and isinstance(v, dict), "Key, value type error"
            self._lengths.append(list(v.keys())[-1])

        first_element: int = self._lengths[0]
        for it in self._lengths:
            if it != first_element:
                assert False, "Sample sizes are not equal"

        self._data: pd.DataFrame = pd.DataFrame.from_dict(feed_dict)
        self._data.set_index("Date", inplace=True)

        self._data_return: pd.DataFrame = self._data.pct_change() * 100
        self._data_return_mask: pd.DataFrame = self._data_return.diff()
        self._data_return_mask: pd.DataFrame = self._data_return_mask.iloc[:, :] > 0

        self._data_return: pd.DataFrame = self._data_return[2:]
        self._data: pd.DataFrame = self._data[2:]
        self._data_return_mask: pd.DataFrame = self._data_return_mask[2:]

    def similar_candle(self, src_index: int) -> Union[Dict[str, Any], None]:
        similar_dates: Dict[str, Any] = self._retrive_similar_date(
            src_index, self._bins, self._data, self._data_return_mask, self._max_samples
        )
        print(f"Date: {similar_dates['base_date']}")

        if len(similar_dates["similar_date"]) > 1:
            # Dict update
            similar_dates: Dict[str, Any] = self._caculate_up_probability(
                self._data, similar_dates, self._fwd
            )
            return similar_dates
        return None

    def _retrive_similar_date(
        self,
        src_idx: int,
        bins: int,
        data: pd.DataFrame,
        data_return_mask: pd.DataFrame,
        max_samples: int = 100,
    ) -> Dict[str, Union[str, List[str], List[float]]]:
        src_data = data.iloc[src_idx - bins : src_idx, :]
        src_mask = data_return_mask.iloc[src_idx - bins : src_idx, :]
        date = src_data.index[-1]

        res = list(
            map(list, self._get_dates(src_data, src_mask, data, data_return_mask, bins))
        )
        res = [it for it in res if it[1] != np.inf]
        if len(res) > 0:
            res.sort(key=lambda x: x[1])

            n_samples = min(len(res), max_samples)
            res = np.array(res[:n_samples])
            score = list(map(float, res[:, 1]))
            return {
                "base_date": date,
                "similar_date": res[:, 0].tolist(),
                "score": score,
            }
        return {}

    def _get_dates(
        self,
        src_data: pd.DataFrame,
        src_mask: pd.DataFrame,
        data: pd.DataFrame,
        data_return_mask: pd.DataFrame,
        bins: int,
    ) -> List[List[Union[str, float]]]:
        return ray.get(
            [
                self._similarity.remote(
                    data.index[idx - 1],
                    src_data,
                    src_mask,
                    data.iloc[idx - bins : idx, :],
                    data_return_mask.iloc[idx - bins : idx, :],
                )
                for idx in range(bins, data.shape[0] + 1)
            ]
        )

    @staticmethod
    @ray.remote(num_cpus=int(0.5 * ps.cpu_count()))
    def _similarity(
        idx: int,
        src: pd.DataFrame,
        src_mask: pd.DataFrame,
        dest: pd.DataFrame,
        dest_mask: pd.DataFrame,
    ) -> Tuple[int, float]:
        """
        Compute the similarity score between two dataframes based on certain conditions.

        Args:
            idx (int): Index of the destination dataframe.
            src (pd.DataFrame): Source dataframe to compare.
            src_mask (pd.DataFrame): Source dataframe's mask.
            dest (pd.DataFrame): Destination dataframe to compare.
            dest_mask (pd.DataFrame): Destination dataframe's mask.

        Returns:
            Tuple[int, float]: A tuple containing the index of the destination dataframe and its similarity score with the source dataframe.
        """
        a_condition = (src_mask.values == dest_mask.values).all()
        b_condition = (
            src.rank(method="min").values == dest.rank(method="min").values
        ).all()
        c_condition = (
            src.rank(method="min", axis=1).values
            == dest.rank(method="min", axis=1).values
        ).all()
        d_condition = (src[-1:]["Close"][0] - src[-2:]["High"][0] > 0) == (
            dest[-1:]["Close"][0] - dest[-2:]["High"][0] > 0
        )
        e_condition = (src[-1:]["Close"][0] - src[-2:]["Low"][0] > 0) == (
            dest[-1:]["Close"][0] - dest[-2:]["Low"][0] > 0
        )

        if a_condition and b_condition and c_condition and d_condition and e_condition:
            x, y = src.values, dest.values
            x_norm = (x - np.min(x)) / (
                np.max(x) - np.min(x)
            )  # normalize source dataframe
            y_norm = (y - np.min(y)) / (
                np.max(y) - np.min(y)
            )  # normalize destination dataframe
            score = np.abs(
                x_norm - y_norm
            ).sum()  # compute the absolute difference between the two normalized dataframes
        else:
            score = np.inf

        return idx, score

    def _caculate_up_probability(
        self, data: pd.DataFrame, similar_dates: Dict[str, Any], fwd: int
    ) -> Dict[str, Any]:
        """
        Calculates the probability of stock prices going up based on the provided data.

        Args:
            data: The dataframe containing the stock prices.
            similar_dates: A dictionary containing information about similar dates.
            fwd: The number of days to look forward in the data.

        Returns:
            A dictionary containing updated information about similar dates, including the probability
            of stock prices going up and the ratio of how many times similar dates occurred in the data.
        """
        similar_date = similar_dates["similar_date"]

        # Get the start and end indices for the forward period
        s_idx = np.sort(np.array(list(map(data.index.get_loc, similar_date))))
        e_idx = s_idx + fwd

        # Remove any indices that are out of bounds
        rm_idx = len(np.argwhere(e_idx > data.shape[0]))
        if rm_idx > 0:
            s_idx = s_idx[:-rm_idx]
            e_idx = e_idx[:-rm_idx]

        # Calculate the probability of stock prices going up
        diff = data.iloc[e_idx]["Close"].values - data.iloc[s_idx]["Close"].values
        up_ratio = np.where(diff > 0, 1, 0).sum() / len(diff)

        # Update the similar_dates dictionary with the up_ratio and occur_ratio
        similar_dates["up_ratio"] = up_ratio
        similar_dates[
            "occur_ratio"
        ] = f"({len(diff)} / {data.shape[0]} {len(diff) / data.shape[0]})"

        return similar_dates

    @property
    def data(self):
        return self._data

    @property
    def data_return(self):
        return self._data_return

    @property
    def data_return_mask(self):
        return self._data_return_mask

    @property
    def bins(self):
        return self._bins

    @property
    def fwd(self):
        return self._fwd

    @property
    def fwd_plot_size(self):
        return self._fwd_plot_size
