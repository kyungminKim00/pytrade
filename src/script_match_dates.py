import os

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ray
import yfinance as yf
from plotly.subplots import make_subplots

ray.init()


# @ray.remote(num_cpus=4)
# def similarity(idx, src, dest):
#     return idx, np.abs(src.values - dest.values).sum()


# @ray.remote(num_cpus=4)
def similarity(idx, src, dest):
    s_mask = src > 0
    d_mask = dest > 0
    if (s_mask.values == d_mask.values).all():
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


def get_dates(recent_data, data_log, bins):
    # return ray.get(
    #     [
    #         similarity.remote(
    #             data_log.index[idx], recent_data, data_log.iloc[idx - 5 : idx, :]
    #         )
    #         for idx in range(bins, data_log.shape[0])
    #     ]
    # )
    return [
        similarity(data_log.index[idx], recent_data, data_log.iloc[idx - 5 : idx, :])
        for idx in range(bins, data_log.shape[0])
    ]


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

    data_log = np.log(data) - np.log(data.shift(1))
    data_log = data_log[1:]
    data = data[1:]
    return data, data_log


def raw_date2(
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

    data_log = pd.DataFrame()
    data_log["Open"] = np.log(data["Close"]) - np.log(data["Open"])
    data_log["High"] = np.log(data["Close"]) - np.log(data["High"])
    data_log["Low"] = np.log(data["Close"]) - np.log(data["Low"])
    data_log["Close"] = 0
    data_log = data_log[1:]
    data = data[1:]
    return data, data_log


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
            )
        ]
    )
    fig1.update_layout(xaxis_rangeslider_visible=False)

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
                )
            ]
        )
        fig2.update_layout(xaxis_rangeslider_visible=False)

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


def retrive_similar_date(src_idx, bins, data_log, max_samples=100):
    src_data = data_log.iloc[src_idx - bins : src_idx, :]
    date = src_data.index[-1]
    res = list(map(list, get_dates(src_data, data_log, bins)))
    res.sort(key=lambda x: x[1])

    # 가장 유사한 max_samples 개
    return {
        "base_date": date,
        "similar_date": np.array(res[:max_samples])[:, 0].tolist(),
    }


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
    # data, data_log = raw_date(
    #     fn="./src/local_data/raw/kospi_1d.csv",
    #     interval="1d",
    #     download=False,
    #     ticker="^KS11",
    # )
    data, data_log = raw_date2(
        fn="./src/local_data/raw/kospi_1d.csv",
        interval="1d",
        download=False,
        ticker="^KS11",
    )

    # 최근 5일 데이터 기준
    bins = 5  # 설정 값
    today_index = data_log.shape[0]  # 설정 값
    fwd = 5  # 설정 값

    # 가장 유사한 100 개
    similar_dates = retrive_similar_date(today_index, bins, data_log, max_samples=5)

    # 확률 값 추가
    similar_dates = caculate_up_probability(data, similar_dates, fwd)

    # plot
    plot_similar_dates(data, similar_dates, bins)
