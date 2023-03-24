import json
import os

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
 
from masked_language_model import MaskedLanguageModel, MaskedLanguageModelDataset
from util import print_c, remove_files

# 모듈 정보
with open("./src/context_prediction.json", "r", encoding="utf-8") as fp:
    env_dict = json.load(fp)


def gather_data(
    predict_up, predict_down, predict_std, predict_up_down, real_rtn, real_std
):
    def is_any(_tensor):
        return (
            torch.concat(_tensor, axis=0).detach().cpu().numpy()
            if len(_tensor) > 0
            else []
        )

    return {
        "predict_up": is_any(predict_up),
        "predict_down": is_any(predict_down),
        "predict_std": is_any(predict_std),
        "predict_up_down": is_any(predict_up_down),
        "real_rtn": is_any(real_rtn),
        "real_std": is_any(real_std),
    }


def to_result(res):
    src = res.detach().cpu().numpy()
    return src


def plot_res(plot_data_train, plot_data_val, plot_data_infer, epoch, loss_type):
    """plot_data_train"""
    y = np.where(plot_data_train["real_rtn"] > 0, 1, 0)
    y_hat = np.where(plot_data_train["predict_up_down"] > 0, 1, 0)
    print(
        f"[train] Accuracy:{accuracy_score(y, y_hat):.3} MAE:{np.mean(np.abs(y - y_hat)):.3}"
    )

    """plot_data_val
    """
    iters = {"valid": plot_data_val, "infer": plot_data_infer}
    for val_or_infer, data_dict in iters.items():
        plot_data = pd.DataFrame.from_dict(data_dict)

        y = np.where(plot_data["real_rtn"] > 0, 1, 0)
        y_hat = np.where(plot_data["predict_up_down"] > 0, 1, 0)

        # # 데이터 Drop
        # plot_data.to_csv(
        #     f"{env_dict['plot_check_dir']}/{loss_type}/prediction/{val_or_infer}_{loss_type}_{epoch}.csv"
        # )

        accuracy = accuracy_score(y, y_hat)
        print(
            f"[{val_or_infer}] Accuracy:{accuracy:.3} MAE:{np.mean(np.abs(y - y_hat)):.3}"
        )

        fig = {}
        colors = px.colors.qualitative.T10  # T10 색상 팔레트 사용
        for i, item in enumerate(
            [
                "predict_up",
                "predict_down",
                "predict_up_down",
                "predict_std",
                "real_rtn",
                "real_std",
            ]
        ):
            fig[item] = go.Scatter(
                x=list(plot_data.index),
                y=plot_data[item],
                line=dict(color=colors[i]),
                legendgroup=item,
                name=item,
            )

        # Create subplots with 1 row and 2 columns
        fig_sub = make_subplots(rows=2, cols=1)
        for item, row in zip(
            ["predict_up", "predict_down", "predict_up_down", "real_rtn"], [1, 1, 1, 1]
        ):
            fig_sub.add_trace(fig[item], row=row, col=1)
        for item, row in zip(["predict_std", "real_std"], [2, 2]):
            fig_sub.add_trace(fig[item], row=row, col=1)

        # Update the subplot layout
        fig_sub.update_layout(title=f"{val_or_infer} accuracy_score: {accuracy:.3}")
        fig_sub.write_image(
            f"{env_dict['plot_check_dir']}/{loss_type}/prediction/{val_or_infer}_{loss_type}_{epoch}_{accuracy:.3}.png"
        )


def plot_animate(loss_type):
    base_dir = env_dict["plot_check_dir"]
    files = []
    for img_fn in os.listdir(base_dir):
        if f"cc_2D_{loss_type}" in img_fn:
            files.append(f"{base_dir}/{loss_type}/{img_fn}")

    for file in files:
        if len(files) > 0:
            images = [imageio.imread(img_fn) for img_fn in file]
            imageio.mimsave(
                f"{base_dir}/{loss_type}/animated_image_{loss_type}.gif",
                images,
                duration=1,
                loop=1,
            )


def Earthmover(y_pred, y_true):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1))
    )


def decode_loss(
    output_vector_up,
    output_vector_down,
    output_vector_std,
    output_vector_up_down,
    fwd,
    up_mask,
    down_mask,
    std_mask,
    criterion,
):
    loss = 0
    val, _ = torch.min(fwd[:, :, :2], dim=-1)
    if up_mask.any():
        loss += criterion(output_vector_up[up_mask], fwd[up_mask][:, 0][:, None])
    if down_mask.any():
        loss += criterion(output_vector_down[down_mask], fwd[down_mask][:, 1][:, None])
    if std_mask.any():
        loss += criterion(output_vector_std[std_mask], fwd[std_mask][:, 2][:, None])
    if down_mask.any():
        loss += criterion(output_vector_up_down[std_mask], val[std_mask][:, None])
    return loss


def learn_model(
    model,
    obs,
    src_mask,
    pad_mask,
    dec_obs,
    dec_src_mask,
    dec_pad_mask,
    fwd,
    up_mask,
    down_mask,
    std_mask,
    criterion,
    domain,
):
    (
        obs,
        src_mask,
        pad_mask,
        fwd,
        dec_obs,
        dec_src_mask,
        dec_pad_mask,
    ) = (
        obs.to(device),
        src_mask.to(device),
        pad_mask.to(device),
        fwd.to(device),
        dec_obs.to(device),
        dec_src_mask.to(device),
        dec_pad_mask.to(device),
    )
    (
        output_vector_up,
        output_vector_down,
        output_vector_std,
        output_vector_up_down,
    ) = model(obs, src_mask, pad_mask, domain, dec_obs, dec_src_mask, dec_pad_mask)
    loss = decode_loss(
        output_vector_up,
        output_vector_down,
        output_vector_std,
        output_vector_up_down,
        fwd,
        up_mask,
        down_mask,
        std_mask,
        criterion,
    )
    return (
        output_vector_up,
        output_vector_down,
        output_vector_std,
        output_vector_up_down,
        loss,
    )


def label_mask(fwd):
    fwd_mask = fwd != np.inf
    up_mask = fwd_mask[:, :, 0]
    down_mask = fwd_mask[:, :, 1]
    std_mask = fwd_mask[:, :, 2]
    return up_mask, down_mask, std_mask


def val_infer(
    dataloader,
    plot_bool,
    model,
    criterion,
    domain,
    predict_up,
    predict_down,
    predict_std,
    predict_up_down,
    real_rtn,
    real_std,
):
    tot_loss = 0
    for (
        obs,
        src_mask,
        pad_mask,
        fwd,
        dec_obs,
        dec_src_mask,
        dec_pad_mask,
    ) in dataloader:
        up_mask, down_mask, std_mask = label_mask(fwd)
        (
            output_vector_up,
            output_vector_down,
            output_vector_std,
            output_vector_up_down,
            loss,
        ) = learn_model(
            model,
            obs,
            src_mask,
            pad_mask,
            dec_obs,
            dec_src_mask,
            dec_pad_mask,
            fwd,
            up_mask,
            down_mask,
            std_mask,
            criterion,
            domain,
        )
        tot_loss += loss.item()

        if plot_bool:
            r_rtn, _ = torch.min(fwd[:, :, :2], dim=-1)
            r_rtn = r_rtn[std_mask]
            r_std = fwd[:, :, -1][std_mask]

            output_vector_up = output_vector_up.squeeze(-1)[std_mask]
            output_vector_down = output_vector_down.squeeze(-1)[std_mask]
            output_vector_std = output_vector_std.squeeze(-1)[std_mask]
            output_vector_up_down = output_vector_up_down.squeeze(-1)[std_mask]

            predict_up.append(output_vector_up[-1][None])
            predict_down.append(output_vector_down[-1][None])
            predict_std.append(output_vector_std[-1][None])
            predict_up_down.append(output_vector_up_down[-1][None])
            real_rtn.append(r_rtn[-1][None])
            real_std.append(r_std[-1][None])
    return (
        predict_up,
        predict_down,
        predict_std,
        predict_up_down,
        real_rtn,
        real_std,
        tot_loss,
    )


def train(
    model,
    train_dataloader,
    val_dataloader,
    infer_dataloader,
    num_epochs,
    lr,
    weight_decay,
    plot_train=False,
    plot_val=False,
    plot_infer=True,
    loss_type="mse",
    domain="band_prediction",
    weight_vars=None,
):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = loss_dict[loss_type]
    best_val = float("inf")
    for epoch in range(num_epochs):
        train_loss = 0
        plot_data_train, plot_data_val = [], []
        predict_up, predict_down, predict_std, predict_up_down, real_rtn, real_std = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        model.train()
        for (
            obs,
            src_mask,
            pad_mask,
            fwd,
            dec_obs,
            dec_src_mask,
            dec_pad_mask,
        ) in train_dataloader:
            up_mask, down_mask, std_mask = label_mask(fwd)
            (
                output_vector_up,
                output_vector_down,
                output_vector_std,
                output_vector_up_down,
                loss,
            ) = learn_model(
                model,
                obs,
                src_mask,
                pad_mask,
                dec_obs,
                dec_src_mask,
                dec_pad_mask,
                fwd,
                up_mask,
                down_mask,
                std_mask,
                criterion,
                domain,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if plot_train:  # gather training results
                r_rtn, _ = torch.min(fwd[:, :, :2], dim=-1)
                r_rtn = r_rtn[std_mask]
                # r_std = fwd[:, :, -1][std_mask]
                # output_vector_up = output_vector_up.squeeze(-1)[std_mask]
                # output_vector_down = output_vector_down.squeeze(-1)[std_mask]
                # output_vector_std = output_vector_std.squeeze(-1)[std_mask]
                output_vector_up_down = output_vector_up_down.squeeze(-1)[std_mask]

                real_rtn.append(r_rtn)
                # real_std.append(r_std)
                # predict_up.append(output_vector_up)
                # predict_down.append(output_vector_down)
                # predict_std.append(output_vector_std)
                predict_up_down.append(output_vector_up_down)

        train_loss /= len(train_dataloader)
        if plot_train:
            plot_data_train = gather_data(
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
            )

        """여기 부터 validation section
        """
        predict_up, predict_down, predict_std, predict_up_down, real_rtn, real_std = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        model.eval()
        with torch.no_grad():
            (
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
                tot_loss,
            ) = val_infer(
                val_dataloader,
                plot_val,
                model,
                criterion,
                domain,
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
            )
        tot_loss /= len(val_dataloader)
        # # Save best model
        # if tot_loss < best_val:
        #     remove_files(env_dict["assets_dir"], loss_type, best_val)
        #     best_val = tot_loss
        #     torch.save(
        #         model.state_dict(),
        #         f"{env_dict['assets_dir']}/{loss_type}_{epoch}_{best_val:.4f}_prediction_model.pt",
        #     )
        torch.save(
            model.state_dict(),
            f"{env_dict['assets_dir']}/{loss_type}_{epoch}_{best_val:.4f}_prediction_model.pt",
        )
        # terminal process - 결과데이터 취합
        if plot_val:
            plot_data_val = gather_data(
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
            )

        """여기 부터 inference section
        """
        predict_up, predict_down, predict_std, predict_up_down, real_rtn, real_std = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        model.eval()
        with torch.no_grad():
            (
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
                _,
            ) = val_infer(
                infer_dataloader,
                plot_infer,
                model,
                criterion,
                domain,
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
            )
        if plot_infer:
            plot_data_infer = gather_data(
                predict_up,
                predict_down,
                predict_std,
                predict_up_down,
                real_rtn,
                real_std,
            )

        # 취합 데이터 plotting
        plot_res(plot_data_train, plot_data_val, plot_data_infer, epoch, loss_type)

        sys_out_print = f"[{loss_type}] Epoch {epoch}: train loss {train_loss*10000:.4f} | val loss {tot_loss*10000:.4f}"
        print(sys_out_print)
        with open(
            f"{env_dict['plot_check_dir']}/{loss_type}/prediction_log.txt",
            "a",
            encoding="utf-8",
        ) as log:
            print(sys_out_print, file=log)


if __name__ == "__main__":
    print("Ray initialized already" if ray.is_initialized() else ray.init())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_dict = {
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "mse": nn.MSELoss(),
        "Earthmover": Earthmover,
        "L1Loss": nn.L1Loss(),
    }
    tv_ratio = env_dict["tv_ratio"]

    print_c("Load dataset")
    cc = load(f"{env_dict['assets_dir']}/crosscorrelation.pkl")
    data = cc.observatoins_merge_idx
    w_vars = cc.weight_variables
    padding_torken = cc.padding_torken
    forward_label = cc.forward_label

    # train configuration
    max_seq_length = env_dict["max_seq_length"]
    batch_size = env_dict["batch_size"]
    # latent size = hidden_size / 4 를 사용하고 있음. Hidden을 너무 적게 유지 할 수 없게 됨 나중에 데이터에 맞게 변경
    hidden_size = env_dict[
        "hidden_size"
    ]  # 1:2 or 1:4 (표현력에 강화), 2:1, 1:1 (특징추출을 변경을 적게 가함)
    step_size = env_dict["step_size"]  # 03.23 58 에폭까지 변화가 없음. lr 줄여봄 (변화없음) -> lr 키워봄
    enable_concept = env_dict["enable_concept"]  # Fix 실험값 변경하지 말기

    num_features = data.shape[1] - 1
    epochs = env_dict["prediction_epochs"]
    print(f"num_features {num_features}")

    # Split data into train and validation sets
    offset = int(len(data) * tv_ratio)
    dist = int((len(data) - offset) * env_dict["val_infer_ratio"])
    train_observations, train_label = data[:offset], forward_label[:offset]
    val_observations, val_label = (
        data[offset : offset + dist],
        forward_label[offset : offset + dist],
    )
    infer_observations, infer_label = (
        data[offset + dist :],
        forward_label[offset + dist :],
    )

    # 데이터 크기 출력
    for k, it in [
        ("Train", train_observations),
        ("Val", val_observations),
        ("Test", infer_observations),
    ]:
        print(f"{k}: {len(it)}")

    # find context model
    model_files = []
    for key, v in loss_dict.items():
        torket = f"{key}_"
        for fn in os.listdir(env_dict["assets_dir"]):
            if torket in fn:
                model_files.append(f"{env_dict['assets_dir']}/{fn}")
    model_files = list(set(model_files))

    # prediciton model 은 mse 고정
    for model_fn in model_files:
        model_mlm = MaskedLanguageModel(
            hidden_size, max_seq_length, num_features, enable_concept
        )
        model_mlm.load_state_dict(torch.load(model_fn))
        model_mlm = model_mlm.to(device)

        # 학습 파라미터 설정
        for name, param in model_mlm.named_parameters():
            if (
                "fc_up" not in name
                and "fc_down" not in name
                and "fc_std" not in name
                and "fc_up_down" not in name
                and "up_down_attn" not in name
                and "fc_up_brg" not in name
                and "fc_down_brg" not in name
                and "decoder_embedding" not in name
                and "transformer_decoder" not in name
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True

        train(
            model=model_mlm,
            train_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    train_observations,
                    max_seq_length,
                    padding_torken=padding_torken,
                    gen_random_mask=False,  # Fix
                    forward_label=train_label,
                    n_period=env_dict["n_period"],
                    mode="decode",
                ),
                batch_size=batch_size,
                shuffle=True,
            ),
            val_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    val_observations,
                    max_seq_length,
                    padding_torken=padding_torken,
                    gen_random_mask=False,  # Fix
                    forward_label=val_label,
                    n_period=env_dict["n_period"],
                    mode="decode",
                ),
                batch_size=1,
                shuffle=False,
            ),
            infer_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    infer_observations,
                    max_seq_length,
                    padding_torken=padding_torken,
                    gen_random_mask=False,  # Fix
                    forward_label=infer_label,
                    mode="decode",
                ),
                batch_size=1,
                shuffle=False,
            ),
            num_epochs=epochs,
            lr=step_size,
            weight_decay=1e-4,
            plot_train=True,
            plot_val=True,
            loss_type="mse",
            domain="band_prediction",
            weight_vars=w_vars,
        )

        # # plot result
        # plot_animate(loss_type=loss_dict["mse"])
