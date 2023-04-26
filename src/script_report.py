import json
import os

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

# 모듈 정보
with open("./src/context_prediction.json", "r", encoding="utf-8") as fp:
    env_dict = json.load(fp)


def plot_animate(data_type):
    base_dir = f"env_dict['plot_check_dir']/img"
    files = []
    for fn in os.listdir(base_dir):
        if f".png" in fn:
            files.append(f"{base_dir}/{fn}")

    if len(files) > 0:
        images = [imageio.imread(fn) for fn in files]
        imageio.mimsave(
            f"{base_dir}/animated_image_{data_type}.gif",
            images,
            duration=0.2,
            loop=1,
        )


if __name__ == "__main__":
    data_type = "infer"
    n_rtn_plot = False
    data = pd.read_csv(env_dict["bias_res"])
    real_rtn = data["real_rtn"]
    real_idx = data["real_idx"]
    prediction_up_down = data["prediction_up_down"]

    y = np.where(real_rtn > 0, 1, 0)
    y_hat = np.where(prediction_up_down > 0, 1, 0)
    img_title = f"Accuracy: {accuracy_score(y, y_hat):.3} MAE: {np.mean(real_rtn-prediction_up_down):.3}"

    alpha = 1
    clip_std = data["predict_std"].clip(-np.inf, 0.06)
    predict_high = prediction_up_down + clip_std * alpha
    predict_low = prediction_up_down - clip_std * alpha

    predict_std = clip_std
    predict_high_idx = (real_idx + (real_idx * predict_high)).shift(60)
    predict_low_idx = (real_idx + (real_idx * predict_low)).shift(60)

    n_samples = real_rtn.shape[0]
    x0 = np.arange(n_samples)
    x1 = x0 + 1
    if n_rtn_plot:
        y0 = predict_low
        y1 = predict_high
        proj_data = pd.DataFrame(
            {
                "proj": real_rtn,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "pred_std": predict_std,
            }
        )
    else:
        y0 = predict_low_idx
        y1 = predict_high_idx
        proj_data = pd.DataFrame(
            {
                "proj": real_rtn,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "pred_std": predict_std,
            }
        )

    # plot data
    canvas = 90
    for idx in range(canvas, n_samples - 2):
        plot_data = proj_data.iloc[idx - canvas : idx]

        x_axis = np.arange(canvas)

        # proj plot
        y_real = proj_data["proj"][: int(canvas * 0.7)].values
        y_dummy = np.array(int(canvas * 0.7) * [np.nan])
        y_real = np.concatenate(y_real, y_dummy)

        trace = go.Scatter(x=x_axis, y=y_real)

        rect_s_idx = int(canvas * 0.8)
        rect_list = []
        rect = plot_data["x0", "x1", "y0", "y1"][rect_s_idx:].values
        pred_std = plot_data["pred_std"][rect_s_idx:].values
        plot_n_rect = canvas - rect_s_idx
        for k in range(plot_n_rect):
            opacity = k / rect.shape[0] + 0.1
            rect_list.append(
                {
                    "type": "rect",
                    "x0": rect_s_idx + k,
                    "x1": rect_s_idx + k + 1,
                    "y0": rect[k, 2],
                    "y1": rect[k, 3],
                    "fillcolor": f"rgba(255, 0, 0, {opacity})",
                    "line": {"width": 0},
                }
            )

        layout = go.Layout(
            title=f"{img_title} std: {np.mean(pred_std)*alpha:.3}",
            shapes=rect_list,
            xaxis=dict(title="days"),
            yaxis=dict(title="index"),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.write_image(f"{env_dict['plot_check_dir']}/img/{idx}.png")

    plot_animate(data_type)
