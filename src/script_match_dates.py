import os

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ray
import yfinance as yf
from plotly.subplots import make_subplots

from util import print_c

ray.shutdown()
ray.init()


# @ray.remote(num_cpus=4)
# def similarity(idx, src, dest):
#     return idx, np.abs(src.values - dest.values).sum()


@ray.remote(num_cpus=4)
def similarity(idx, src, src_mask, dest, dest_mask):
    a_condition = (src_mask.values == dest_mask.values).all()
    b_condition = (
        src.rank(method="min").values == dest.rank(method="min").values
    ).all()
    c_condition = (
        src.rank(method="min", axis=1).values == dest.rank(method="min", axis=1).values
    ).all()

    # # 오리지널 버젼
    # if a_condition and b_condition and c_condition:
    #     score = np.abs(src.values - dest.values).sum()
    # else:
    #     score = np.inf

    if a_condition and b_condition:
        score = np.abs(src.values - dest.values).sum()
    else:
        score = np.inf
    return idx, score


def plot_animate(loss_type):
    base_dir = "./src/local_data/assets/plot_check/chart"
    files = []
    for fn in os.listdir(base_dir):
        if f"cc_2D_{loss_type}" in fn:
            files.append(f"{base_dir}/{loss_type}/{fn}")

    for file in files:
        if len(files) > 0:
            images = [imageio.imread(fn) for fn in file]
            imageio.mimsave(
                f"{base_dir}/{loss_type}/animated_image_{loss_type}.gif",
                images,
                duration=1,
                loop=1,
            )


def get_dates(src_data, src_mask, data_return, data_return_mask, bins):
    return ray.get(
        [
            similarity.remote(
                data_return.index[idx - 1],
                src_data,
                src_mask,
                data_return.iloc[idx - 5 : idx, :],
                data_return_mask.iloc[idx - 5 : idx, :],
            )
            for idx in range(bins, data_return.shape[0] + 1)
        ]
    )
    # return [
    #     similarity(
    #         data_return.index[idx - 1],
    #         src_data,
    #         src_mask,
    #         data_return.iloc[idx - 5 : idx, :],
    #         data_return_mask.iloc[idx - 5 : idx, :],
    #     )
    #     for idx in range(bins, data_return.shape[0] + 1)
    # ]


def raw_date(
    fn="./src/local_data/raw/kospi_1d.csv",
    interval="1d",
    download=False,
    ticker="^KS11",
):
    if download:
        # KOSPI - Download
        ticker = ticker
        data = yf.download(tickers=ticker, period="max", interval=interval)
        pd.DataFrame(data).to_csv(fn)
    # Data Load
    data = pd.read_csv(fn)
    data.set_index("Date", inplace=True)
    data.drop(columns=["Adj Close", "Volume"], inplace=True)

    data_return = data.pct_change() * 100
    data_return_mask = data_return.diff()
    data_return_mask = data_return_mask.iloc[:, :] > 0

    data_return = data_return[2:]
    data = data[2:]
    data_return_mask = data_return_mask[2:]
    return data, data_return, data_return_mask


def plot_similar_dates(data, similar_dates, bins):
    base_date = similar_dates["base_date"]
    similar_date = similar_dates["similar_date"]
    up_ratio = similar_dates["up_ratio"]

    base_idx = data.index.get_loc(base_date)
    target_idx = np.array(list(map(data.index.get_loc, similar_date)))

    base_data = data.iloc[base_idx - bins : base_idx]

    fig1 = go.Figure(
        data=[
            go.Candlestick(
                x=np.arange(base_data.shape[0]),
                open=base_data["Open"],
                high=base_data["High"],
                low=base_data["Low"],
                close=base_data["Close"],
                increasing="red",  # 상승봉
                decreasing="blue",  # 하락봉
            )
        ]
    )
    fig1.update_layout(xaxis_rangeslider_visible=False)
    fig1.show()

    for f_ord, t_idx in enumerate(target_idx.tolist()):
        target_data = data.iloc[t_idx - bins : t_idx]
        fig2 = go.Figure(
            data=[
                go.Candlestick(
                    x=np.arange(target_data.shape[0]),
                    open=target_data["Open"],
                    high=target_data["High"],
                    low=target_data["Low"],
                    close=target_data["Close"],
                    increasing="red",  # 상승봉
                    decreasing="blue",  # 하락봉
                )
            ]
        )
        fig2.update_layout(xaxis_rangeslider_visible=False)
        fig2.show()

        # Create subplots with 1 row and 2 columns
        fig_sub = make_subplots(rows=1, cols=2)

        # Add the two figures to the subplot
        fig_sub.add_trace(fig1.data[0], row=1, col=1)
        fig_sub.add_trace(fig2.data[0], row=1, col=2)

        # Update the subplot layout
        fig_sub.update_layout(title=f"Date: {base_date} up: {up_ratio}")

        fig_sub.write_image(
            f"./src/local_data/assets/plot_check/chart/{f_ord}_{base_date}_{target_data.index[-1]}.png"
        )


def retrive_similar_date(src_idx, bins, data_return, data_return_mask, max_samples=100):
    src_data = data_return.iloc[src_idx - bins : src_idx, :]
    src_mask = data_return_mask.iloc[src_idx - bins : src_idx, :]
    date = src_data.index[-1]
    res = list(
        map(list, get_dates(src_data, src_mask, data_return, data_return_mask, bins))
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


def caculate_up_probability(data, similar_dates, fwd):
    similar_date = similar_dates["similar_date"]

    s_idx = np.sort(np.array(list(map(data.index.get_loc, similar_date))))
    e_idx = s_idx + fwd

    rm_idx = len(np.argwhere(e_idx > data.shape[0]))
    s_idx = s_idx[:-rm_idx]
    e_idx = e_idx[:-rm_idx]

    diff = data.iloc[e_idx]["Close"].values - data.iloc[s_idx]["Close"].values
    up_ratio = np.where(diff > 0, 1, 0).sum() / len(diff)

    similar_dates["up_ratio"] = up_ratio

    return similar_dates


if __name__ == "__main__":
    # # 데이터 다운로드
    # data, data_return = raw_date(
    #     fn="./src/local_data/raw/kospi_1d.csv",
    #     interval="1d",
    #     download=False,
    #     ticker="^KS11",
    # )
    data, data_return, data_return_mask = raw_date(
        fn="./src/local_data/raw/kospi_1d.csv",
        interval="1d",
        download=False,
        ticker="^KS11",
    )

    # 최근 5일 데이터 기준
    bins = 5  # 설정 값
    today_index = data_return.shape[0]  # 설정 값
    fwd = 5  # 설정 값

    for today_index in range(bins, today_index + 1):
        similar_dates = retrive_similar_date(
            today_index, bins, data_return, data_return_mask, max_samples=5
        )
        print_c(f"Date: {similar_dates['base_date']}")
        if len(similar_dates["similar_date"]) > 1:
            # 확률 값 추가
            similar_dates = caculate_up_probability(data, similar_dates, fwd)
            # plot
            plot_similar_dates(data, similar_dates, bins)
