import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ray
import yfinance as yf
from plotly.subplots import make_subplots

from candle_stick import CandleStick

print("Ray initialized already" if ray.is_initialized() else ray.init())


# 실험용 플롯 및 디버그용 Plot 코드 삭제 하지 말 것
def plot_similar_dates(data, similar_dates, bins, fwd, fwd_plot_size, img_dir):
    base_date = similar_dates["base_date"]
    similar_date = similar_dates["similar_date"]
    up_ratio = similar_dates["up_ratio"]
    occur_ratio = similar_dates["occur_ratio"]

    base_idx = data.index.get_loc(base_date) + 1  # [a:b] 는 b index 앞까지의 데이터
    target_idx = (
        np.array(list(map(data.index.get_loc, similar_date))) + 1
    )  # [a:b] 는 b index 앞까지의 데이터

    base_data = data.iloc[base_idx - bins : base_idx + fwd_plot_size]

    # 왼쪽 그림
    fig1 = go.Figure(
        data=[
            go.Candlestick(
                x=np.arange(base_data.shape[0]),
                open=base_data["Open"],
                high=base_data["High"],
                low=base_data["Low"],
                close=base_data["Close"],
                increasing_line_color="red",  # 상승봉
                decreasing_line_color="blue",  # 하락봉
            )
        ]
    )
    fig1.update_layout(
        xaxis=dict(rangeslider=dict(visible=False)),
        shapes=[
            dict(
                x0=0,
                x1=bins - 1,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor="green",
                opacity=0.25,
                line_width=0,
                layer="below",
            ),
            dict(
                x0=bins - 1,
                x1=bins - 1 + fwd,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor="red",
                opacity=0.25,
                line_width=0,
                layer="below",
            ),
        ],
    )

    for f_ord, t_idx in enumerate(target_idx.tolist()):
        target_data = data.iloc[t_idx - bins : t_idx + fwd_plot_size]
        # 오른쪽 그림
        fig2 = go.Figure(
            data=[
                go.Candlestick(
                    x=np.arange(target_data.shape[0]),
                    open=target_data["Open"],
                    high=target_data["High"],
                    low=target_data["Low"],
                    close=target_data["Close"],
                    increasing_line_color="red",  # 상승봉
                    decreasing_line_color="blue",  # 하락봉
                )
            ]
        )
        fig2.update_layout(
            xaxis=dict(rangeslider=dict(visible=False)),
            shapes=[
                dict(
                    x0=0,
                    x1=bins - 1,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="green",
                    opacity=0.25,
                    line_width=0,
                    layer="below",
                ),
                dict(
                    x0=bins - 1,
                    x1=bins - 1 + fwd,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="red",
                    opacity=0.25,
                    line_width=0,
                    layer="below",
                ),
            ],
        )
        # Create subplots with 1 row and 2 columns
        fig_sub = make_subplots(rows=1, cols=2)

        # Add the two figures data to the subplot
        fig_sub.add_trace(fig1.data[0], row=1, col=1)
        fig_sub.add_trace(fig2.data[0], row=1, col=2)

        fig_sub.update_layout(
            title=f"Date: {base_date} up: {up_ratio} occur: {occur_ratio}",
            shapes=fig1.layout.shapes + fig2.layout.shapes,
        )

        if base_date != target_data.index[-1]:
            if len(target_idx) > 2:  # min samples
                fig_sub.write_image(
                    f"{img_dir}/B{bins}_{base_date}_{f_ord}_{target_data.index[-1]}.png"
                )


def raw_date(
    fn="./src/local_data/raw/kospi_1d.csv",
    interval="1d",
    download=False,
    ticker="^KS11",
):
    # KOSPI - Download
    if download:
        pd.DataFrame(
            yf.download(tickers=ticker, period="max", interval=interval)
        ).to_csv(fn)
    return pd.read_csv(fn).to_dict()


if __name__ == "__main__":
    # 모듈 정보
    with open("./src/candle_stick.json", "r", encoding="utf-8") as fp:
        env_dict = json.load(fp)

    # 데이터 다운로드 and 로드
    raw = raw_date(
        fn=env_dict["raw_file_csv"],
        interval=env_dict["interval"],
        download=False,
        ticker=env_dict["ticker"],
    )

    """이 모듈은 전체 시계의 데이터를 Dictionary 형태로 입력 받을 것을 가정 함
    """
    # 입력 데이터 형태
    src_data = {
        "Date": raw["Date"],
        "Open": raw["Open"],
        "High": raw["High"],
        "Low": raw["Low"],
        "Close": raw["Close"],
    }

    debug = True
    for n_candles in [3, 4, 5]:
        # 파라미터 설정
        cs = CandleStick(
            feed_dict=src_data,
            bins=n_candles,
            fwd=env_dict["fwd"],
            fwd_plot_size=env_dict["fwd_plot_size"],
            max_samples=env_dict["max_samples"],
        )

        for src_index in range(n_candles, cs.data.shape[0] + 1):
            # 데이터 인텍스의 데이터와 비슷한 캔들의 날짜를 리턴
            res_similar_dates = cs.similar_candle(src_index)

            # plot - 디버그용
            if res_similar_dates is not None and debug:
                plot_similar_dates(
                    cs.data,
                    res_similar_dates,
                    cs.bins,
                    cs.fwd,
                    cs.fwd_plot_size,
                    env_dict["debug_plot_dir"],
                )
