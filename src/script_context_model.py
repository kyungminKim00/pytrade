import json
import os

import imageio
import numpy as np
import plotly.express as px
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from cross_correlation import CrossCorrelation
from masked_language_model import MaskedLanguageModel, MaskedLanguageModelDataset
from util import print_c, remove_files


def to_result(res, label=0):
    src = res.detach().cpu().numpy()
    return np.concatenate(
        (
            src,
            (np.ones([src.shape[0], 1]) * label).astype(np.int32),
        ),
        axis=1,
    )


def plot_res(plot_data_train, plot_data_val, epoch, loss_type):
    fig = []
    for plot_data in [plot_data_train, plot_data_val]:
        fig.append(
            px.scatter(
                plot_data,
                x=plot_data[:, 0],
                y=plot_data[:, 1],
                color=plot_data[:, -1],
                title="2D visualization for cross-correlation (blue: real yellow: predict)",
            )
        )

    # Create subplots with 1 row and 2 columns
    fig_sub = make_subplots(rows=1, cols=2)

    # Add the two figures to the subplot
    fig_sub.add_trace(fig[0].data[0], row=1, col=1)
    fig_sub.add_trace(fig[1].data[0], row=1, col=2)

    # Update the subplot layout
    fig_sub.update_layout(
        title="2D visualization for cross-correlation (blue: real yellow: predict)"
    )

    fig_sub.write_image(
        f"./src/local_data/assets/plot_check/{loss_type}/cc_2D_{loss_type}_{epoch}.png"
    )


def plot_animate(loss_type):
    base_dir = "./src/local_data/assets/plot_check"
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


def Earthmover(y_pred, y_true):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1))
    )


# def KL_divergence(mu, log_var):
#     # 시퀀스의 합인거 같은데 이상하네, 배치단위로 합을 취하고 평균을 구할때 동작 함
#     # 정규분포를 가정하더라도 이는 시퀀스 별 분포가 매우 다르기 때문에 발생하는 현상으로 생각 됨
#     var = torch.exp(log_var)
#     kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - var, dim=1).mean()
#     return kl_loss


def KL_divergence(z, mu, log_var):
    var = torch.exp(0.5 * log_var)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(var))
    q = torch.distributions.Normal(mu, var)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    kl_loss = log_qzx - log_pz
    kl_loss = kl_loss.sum(dim=-1)
    return kl_loss.mean()


def cal_density(data):
    # 2 components 만 활용 - 시각화 해석에 용이
    diff = data.unsqueeze(1)[:, :, :2] - data.unsqueeze(0)[:, :, :2]
    return (
        torch.sqrt(torch.sum(diff**2, axis=-1)).sum(axis=0)[:, None] / data.shape[0]
    )


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    lr,
    weight_decay,
    plot_train=False,
    plot_val=False,
    loss_type="mse",
    domain="context",
    weight_vars=None,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = loss_dict[loss_type]
    best_val_density = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        tmp_data, plot_data_train, plot_data_val = [], [], []
        for obs, masked_obs, src_mask, pad_mask, _, _, _, _ in train_dataloader:
            obs, src_mask, pad_mask, masked_obs = (
                obs.to(device),
                src_mask.to(device),
                pad_mask.to(device),
                masked_obs.to(device),
            )
            contain_blank = ~src_mask
            if contain_blank.sum() > 0:
                output, mean, log_var, z = model(masked_obs, src_mask, pad_mask, domain)
                # Calculate loss only for masked tokens
                loss = criterion(
                    output[contain_blank], obs[contain_blank]
                ) + KL_divergence(z, mean, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if plot_train:
                    tmp_data.append(to_result(output[contain_blank], label=1))
                    tmp_data.append(to_result(obs[contain_blank], label=0))
        train_loss /= len(train_dataloader)
        if plot_train:
            plot_data_train = np.concatenate(tmp_data, axis=0)

        model.eval()
        val_loss = 0
        dist = 0
        tmp_data = []
        density_output, density_obs = [], []
        with torch.no_grad():
            for obs, masked_obs, src_mask, pad_mask, _, _, _, _ in val_dataloader:
                obs, src_mask, pad_mask, masked_obs = (
                    obs.to(device),
                    src_mask.to(device),
                    pad_mask.to(device),
                    masked_obs.to(device),
                )
                contain_blank = ~src_mask
                if contain_blank.sum() > 0:
                    output, mean, log_var, z = model(
                        masked_obs, src_mask, pad_mask, domain
                    )
                    loss = criterion(
                        output[contain_blank], obs[contain_blank]
                    ) + KL_divergence(z, mean, log_var)
                    val_loss += loss.item()

                    dist += torch.abs(output[contain_blank] - obs[contain_blank]).sum()
                    density_output.append(output[contain_blank])
                    density_obs.append(obs[contain_blank])

                    if plot_val:
                        tmp_data.append(to_result(output[contain_blank], label=1))
                        tmp_data.append(to_result(obs[contain_blank], label=0))
        val_loss /= len(val_dataloader)
        dist /= len(val_dataloader)

        density = []
        if len(density_output) > 0:
            density_output = torch.cat(density_output, dim=0)
            density_obs = torch.cat(density_obs, dim=0)

            for d_data in [density_output, density_obs]:
                density.append(cal_density(d_data).mean())
            density = torch.abs(density[1] - density[0])

        # terminal process
        if plot_val:
            plot_data_val = np.concatenate(tmp_data, axis=0)
            plot_res(plot_data_train, plot_data_val, epoch, loss_type)

        sys_out_print = f"[{loss_type}] Epoch {epoch}: train loss {train_loss*10000:.4f} | val loss {val_loss*10000:.4f} | dist {dist:.4f} | density {density:.4f}"
        print(sys_out_print)
        with open(
            f"./src/local_data/assets/plot_check/{loss_type}/log.txt",
            "a",
            encoding="utf-8",
        ) as log:
            print(sys_out_print, file=log)

        # # Save best model
        # if density < best_val_density:
        #     best_val_density = density
        #     remove_files("./src/local_data/assets", loss_type, density)
        #     torch.save(
        #         model.state_dict(),
        #         f"./src/local_data/assets/{loss_type}_{epoch}_{density:.4f}_context_model.pt",
        #     )
        torch.save(
            model.state_dict(),
            f"./src/local_data/assets/{loss_type}_{epoch}_{density:.4f}_context_model.pt",
        )


if __name__ == "__main__":
    print("Ray initialized already" if ray.is_initialized() else ray.init())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모듈 정보
    with open("./src/context_prediction.json", "r", encoding="utf-8") as fp:
        env_dict = json.load(fp)

    loss_dict = {
        # "SmoothL1Loss": nn.SmoothL1Loss(),
        "mse": nn.MSELoss(),
        # "Earthmover": Earthmover,
        # "L1Loss": nn.L1Loss(),
    }

    tv_ratio = env_dict["tv_ratio"]
    # 특징 추출
    cc = CrossCorrelation(
        mv_bin=20,  # bin size for moving variance
        correation_bin=60,  # bin size to calculate cross correlation
        x_file_name="./src/local_data/raw/x_toys.csv",
        y_file_name="./src/local_data/raw/y_toys.csv",
        debug=False,
        data_tranform=None,  # None: 데이터변환 수행안함, n_components: np.inf 는 전체 차원
        # data_tranform={
        #     "n_components": 8,
        #     "method": "UMAP",
        # },  # None: 데이터변환 수행안함, n_components: np.inf 는 전체 차원
        ratio=tv_ratio,  # data_tranform is not None 일 때 PCA 학습 샘플, 차원 축소된 observation 은 전체 샘플 다 포함 함
        alpha=1.5,
    )
    dump(cc, "./src/local_data/assets/crosscorrelation.pkl")
    cc = load("./src/local_data/assets/crosscorrelation.pkl")
    data = cc.observatoins_merge_idx
    weight_vars = cc.weight_variables
    padding_torken = cc.padding_torken
    forward_label = cc.forward_label

    # train configuration
    max_seq_length = env_dict["max_seq_length"]
    batch_size = env_dict["batch_size"]
    # latent size = hidden_size / 4 를 사용하고 있음. Hidden을 너무 적게 유지 할 수 없게 됨 나중에 데이터에 맞게 변경
    hidden_size = env_dict[
        "hidden_size"
    ]  # 1:2 or 1:4 (표현력에 강화), 2:1, 1:1 (특징추출을 변경을 적게 가함)
    step_size = env_dict["step_size"]
    enable_concept = env_dict["enable_concept"]  # Fix 실험값 변경하지 말기

    num_features = data.shape[1] - 1
    epochs = 100  # 150
    print(f"num_features {num_features}")

    # Split data into train and validation sets
    offset = int(len(data) * tv_ratio)
    dist = int((len(data) - offset) * 0.5)
    train_observations, train_label = data[:offset], forward_label[:offset]
    val_observations, val_label = (
        data[offset : offset + dist],
        forward_label[offset : offset + dist],
    )
    test_observations, test_label = (
        data[offset + dist :],
        forward_label[offset + dist :],
    )

    # 데이터 크기 출력
    for k, it in [
        ("Train", train_observations),
        ("Val", val_observations),
        ("Test", test_observations),
    ]:
        print(f"{k}: {len(it)}")

    # context model training
    for k, v in loss_dict.items():
        train(
            model=MaskedLanguageModel(
                hidden_size, max_seq_length, num_features, enable_concept
            ).to(device),
            train_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    train_observations,
                    max_seq_length,
                    padding_torken=padding_torken,
                    gen_random_mask=True,  # Fix
                    forward_label=train_label,
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
                ),
                batch_size=batch_size,
                shuffle=False,
            ),
            num_epochs=epochs,
            lr=step_size,
            weight_decay=1e-4,
            plot_train=True,
            plot_val=True,
            loss_type=k,
            domain="context",
            weight_vars=weight_vars,
        )

        # plot result
        plot_animate(loss_type=k)
